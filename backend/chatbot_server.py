import os
import time
import requests
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_weaviate.vectorstores import WeaviateVectorStore
from langchain.docstore.document import Document
from langfuse import Langfuse
from langfuse.decorators import observe
from weaviate.client import WeaviateClient
from weaviate.connect import ConnectionParams
from weaviate.classes.query import Filter

# === Load environment ===
load_dotenv()

# === Config ===
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Setup logging  ===
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# === Langfuse setup ===
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    debug=True,
)

try:
    response = requests.get(f"{os.getenv('LANGFUSE_HOST')}/api/public/health")
    if response.status_code == 200:
        print("✅ Langfuse server is healthy:", response.json())
    else:
        print(f"⚠️ Langfuse health endpoint returned status: {response.status_code}")
except Exception as e:
    print(f"❌ Failed to reach Langfuse server: {e}")


# === Wait for Weaviate ===
def wait_for_weaviate(url, timeout=30):
    print(f"⏳ Waiting for Weaviate at {url}...")
    for _ in range(timeout):
        try:
            if requests.get(f"{url}/v1/.well-known/ready").status_code == 200:
                print("✅ Weaviate is ready")
                return
        except:
            pass
        time.sleep(1)
    raise Exception("Weaviate did not start in time")


wait_for_weaviate(f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}")

# === FastAPI setup ===
app = FastAPI()

# === Weaviate setup ===
connection_params = ConnectionParams.from_params(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_PORT,
    http_secure=False,
    grpc_host="weaviate",
    grpc_port=50051,
    grpc_secure=False,
)
client = WeaviateClient(connection_params=connection_params)
client.connect()

print("🧪 Verifying index structure:")
print(client.collections.list_all())
print("🧪 Sample object:")
print(client.collections.get("LangchainDocs").query.fetch_objects(limit=1))

embedding = OpenAIEmbeddings()
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="LangchainDocs",
    text_key="text",
    embedding=embedding,
)

retriever = vectorstore.as_retriever()


# === Tool definition ===
# Use Weaviate native client to fetch neighbors


def fetch_neighbor_chunk(filename, chunk_idx):
    try:
        filters = (
                Filter.by_property("filename").equal(filename) &
                Filter.by_property("chunk_index").equal(int(chunk_idx))
        )

        result = client.collections.get("LangchainDocs").query.fetch_objects(
            filters=filters,
            limit=1,
            return_properties=["text", "filename", "chunk_index"]
        )
        print(f"🔍   Found {len(result.objects)} objects for chunk {chunk_idx} of {filename}")

        return [
            Document(
                page_content=obj.properties["text"],
                metadata={
                    "filename": obj.properties.get("filename"),
                    "chunk_index": obj.properties.get("chunk_index")
                }
            )
            for obj in result.objects
        ]

    except Exception as e:
        print(f"⚠️ Error fetching chunk {chunk_idx} for {filename}: {e}")
        return []


@tool
def search_docs(query: str, n=3):
    """Search documentation and return each match with 1 chunk before and after."""
    print(f"🔍 search_docs called with query: {query}")
    docs = retriever.invoke(query)

    context_chunks = []
    seen = set()

    for doc in docs:

        print(f'🔍   Found match: {doc.metadata.get("chunk_index")}  {doc.page_content}')
        filename = doc.metadata.get("filename")
        idx = doc.metadata.get("chunk_index")

        # Add current match
        if doc.page_content not in seen:
            context_chunks.append(doc)
            seen.add(doc.page_content)

        # Try to add surrounding chunks
        for offset in [-1, 1]:
            neighbor_idx = idx + offset
            neighbors = fetch_neighbor_chunk(filename, neighbor_idx)
            print("🔍   Number of neighbors found:", len(neighbors))
            for neighbor in neighbors:
                if neighbor.page_content not in seen:
                    context_chunks.append(neighbor)
                    seen.add(neighbor.page_content)

    # Format and return
    result = "\n\n".join(doc.page_content for doc in context_chunks[:n])
    print(f"📚 search_docs result:\n{result}")
    print(f"all docs found {len(context_chunks)}, n docs returned {len(context_chunks[:n])}.\n")
    logger.info(
        f"all docs found {len(docs)}, n unique {len(context_chunks)} n docs returned {len(context_chunks[:n])}.\n")
    logger.info(print(f"📚 search_docs result:\n{result}"))
    return result


# === Prompt with memory ===
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Use search_docs to answer questions. If answer is not in documents, return that. Remember user's name if mentioned. Use chat history to answer contextually."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# === Bind tools to model ===
llm = ChatOpenAI(model="gpt-4", temperature=0).bind_tools([search_docs])

store = {}


def get_memory(session_id):
    return store.setdefault(session_id, ChatMessageHistory())


chat_chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=get_memory,
    input_messages_key="input",
    history_messages_key="history"
)


# === Observed Chat Chain Call ===
@observe(name="chat_chain_invoke")
async def invoke_chat_chain(user_input: str, session_id: str, trace_id: str):
    # manually pass trace_id
    response = await chat_chain.ainvoke(
        {"input": user_input},
        config={
            "configurable": {
                "session_id": session_id,
                "langfuse": {"trace_id": trace_id}
            }
        },
    )
    return response


# === FastAPI schema ===
class ChatRequest(BaseModel):
    user_id: str
    message: str


# === Chat Endpoint ===
@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.user_id
    user_input = req.message

    logger.info(f"[{session_id}] 📨 User input: {user_input}")

    # 🔥 Start Langfuse trace

    # manually start trace
    trace = langfuse.trace(
        name="chat_session",
        user_id=session_id,
        metadata={"user_message": user_input},
    )

    try:
        # First model call
        response = await invoke_chat_chain(
            user_input=user_input,
            session_id=session_id,
            trace_id=trace.id,  # <<< important!
        )
        logger.info(f"[{session_id}] 🤖 First response: {response.content}")

        if response.tool_calls:
            logger.info(f"[{session_id}] 🛠 Tool calls: {response.tool_calls}")
            memory = get_memory(session_id)
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name == "search_docs":
                    try:
                        tool_result = search_docs.invoke(tool_args)
                        logger.info(f"[{session_id}] 📚 Tool result: {tool_result}")
                        trace.span(
                            name="search_docs",
                            input=tool_args,
                            output=tool_result,
                            metadata={"tool_call_id": tool_id}
                        )
                        memory.add_message(ToolMessage(tool_call_id=tool_id, content=tool_result))

                    except Exception as tool_err:
                        logger.exception(f"[{session_id}] ❌ Tool error: {str(tool_err)}")
                        return {"response": "Something went wrong during document search."}

            # Second model call after tool
            final_response = await invoke_chat_chain(user_input="", session_id=session_id, trace_id=trace.id)
            logger.info(f"[{session_id}] 🤖 Final response: {final_response.content}")

            return {"response": final_response.content}

        else:
            # No tools were used
            return {"response": response.content}

    except Exception as e:
        logger.exception(f"[{session_id}] ❌ Error: {str(e)}")
        return {"response": "Something went wrong."}


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.on_event("shutdown")
def shutdown_event():
    client.close()
    print("✅ Weaviate client closed on shutdown")
