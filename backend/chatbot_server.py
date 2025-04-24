import os
import time
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.client import WeaviateClient
from weaviate.connect import ConnectionParams

from langchain_core.documents import Document

# === Load environment ===
load_dotenv()

# === Config ===
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# === Wait for Weaviate ===
def wait_for_weaviate(url, timeout=10):
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
print(client.collections.get("LangChain").query.fetch_objects(limit=1))

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

# Re-query after inserting
results = retriever.invoke("advokatuur")
print("🔍 Manual Weaviate test result:", results)

# === Tool definition ===
@tool
def search_docs(query: str) -> str:
    """Search documentation for relevant content."""
    print(f"🔍 search_docs called with query: {query}")
    docs = retriever.invoke(query)
    result = "\n\n".join([doc.page_content for doc in docs[:3]])
    print(f"📚 search_docs result:\n{result}")
    return result

# === Prompt with memory placeholder ===
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use search_docs to answer questions. If answer is not in documents, return that. Remember user's name if mentioned. Use chat history to answer contextually."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# === Bind tools to model ===
llm = ChatOpenAI(model="gpt-4", temperature=0).bind_tools([search_docs])

# === Store session memory ===
store = {}

def get_memory(session_id):
    return store.setdefault(session_id, ChatMessageHistory())

# === Chain with memory ===
chat_chain = RunnableWithMessageHistory(
    prompt | llm,
    get_session_history=get_memory,
    input_messages_key="input",
    history_messages_key="history"
)

# === FastAPI schema ===
class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    session_id = req.user_id
    user_input = req.message

    # First round: get assistant response
    response = await chat_chain.ainvoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    # If tool call exists, resolve it and call again
    if response.tool_calls:
        tool_messages = []

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_id = tool_call["id"]

            if tool_name == "search_docs":
                tool_result = search_docs.invoke(tool_args)

                # Append tool response to memory directly
                memory = get_memory(session_id)
                memory.add_message(ToolMessage(tool_call_id=tool_id, content=tool_result))

        # Re-run chain with the tool response now in memory
        final_response = await chat_chain.ainvoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        return {"response": final_response.content}

    return {"response": response.content}

@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()