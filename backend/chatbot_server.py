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
from langfuse import Langfuse
from langfuse.decorators import observe
from weaviate.client import WeaviateClient
from weaviate.connect import ConnectionParams

from langchain_core.documents import Document

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

from langfuse import Langfuse
#TODO
langfuse = Langfuse(
  secret_key="sk-lf-edcc7119-bc4a-4c76-9f8c-515bf7e8c238",
  public_key="pk-lf-567548a7-51ee-4bfd-a488-cfdfdb00ce8f",
  host="http://localhost:3000",
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

print("Testing Weaviate query:")
print(retriever.invoke("advokatuur"))

# Test inserting a document
test_doc = Document(page_content="Advokatuur on Eesti õigussüsteemi osa.")
vectorstore.add_documents([test_doc])

results = retriever.invoke("advokatuur")
print("🔍 Manual Weaviate test result:", results)

# === Tool definition ===
@tool
@observe(name="search_docs_tool")  # This will appear as a separate trace span in Langfuse
def search_docs(query: str) -> str:
    """Search documentation for relevant content."""
    print(f"🔍 search_docs called with query: {query}")
    docs = retriever.invoke(query)
    seen = set()
    unique_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique_docs.append(content)
    result = "\n\n".join(unique_docs[:3])
    print(f"📚 search_docs result:\n{result}")
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
async def invoke_chat_chain(user_input: str, session_id: str, trace=None):
    try:
        response = await chat_chain.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        # Optional manual logging inside this observation
        if trace:
            trace.generation(
                name="model_response",
                input=user_input,
                output=response.content
            )
            trace.score(name="chain_success", value=1)

        return response

    except Exception as e:
        if trace:
            trace.score(name="chain_success", value=0)
            trace.generation(
                name="error",
                input=user_input,
                output=str(e)
            )
        raise e
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
    trace = langfuse.trace(
        name="chat_session",
        user_id=session_id,
        metadata={"user_message": user_input},
    )
    user_input_gen = trace.generation(
        name="user_message",
        input=user_input,
        output="(processing)",
    )

    try:
        # First model call
        response = await invoke_chat_chain(user_input=user_input, session_id=session_id, trace=trace)
        logger.info(f"[{session_id}] 🤖 First response: {response.content}")

        if response.tool_calls:
            logger.info(f"[{session_id}] 🛠 Tool calls: {response.tool_calls}")

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                if tool_name == "search_docs":
                    tool_span = trace.span(
                        name="search_docs_tool",
                        input=tool_args,
                        metadata={"tool_call_id": tool_id},
                    )
                    try:
                        tool_result = search_docs.invoke(tool_args)
                        tool_span.end(output=tool_result)

                        memory = get_memory(session_id)
                        memory.add_message(ToolMessage(tool_call_id=tool_id, content=tool_result))

                    except Exception as tool_err:
                        tool_span.end(output=str(tool_err), level="ERROR")
                        logger.exception(f"[{session_id}] ❌ Tool error: {str(tool_err)}")
                        user_input_gen.end(output="Tool failed")
                        trace.score(name="conversation_success", value=0)
                        return {"response": "Something went wrong during document search."}

            # Second model call after tool
            final_response = await invoke_chat_chain(user_input=user_input, session_id=session_id, trace=trace)
            logger.info(f"[{session_id}] 🤖 Final response: {final_response.content}")

            user_input_gen.end(output=final_response.content)
            trace.score(name="conversation_success", value=1)

            return {"response": final_response.content}

        else:
            # No tools were used
            user_input_gen.end(output=response.content)
            trace.score(name="conversation_success", value=1)
            return {"response": response.content}

    except Exception as e:
        logger.exception(f"[{session_id}] ❌ Error: {str(e)}")
        user_input_gen.end(output="Error")
        trace.score(name="conversation_success", value=0)
        return {"response": "Something went wrong."}


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.on_event("shutdown")
def shutdown_event():
    client.close()
    print("✅ Weaviate client closed on shutdown")
