import os
import time
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain.agents import Tool, AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.client import WeaviateClient
from weaviate.connect import ConnectionParams

load_dotenv()

# === Config ===
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST")
WEAVIATE_PORT = os.getenv("WEAVIATE_PORT")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# === Wait for Weaviate to be ready ===
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


wait_for_weaviate(f'http://{WEAVIATE_HOST}:{WEAVIATE_PORT}')

# === FastAPI setup ===
app = FastAPI()

# === Weaviate client v4 ===
connection_params = ConnectionParams.from_params(
    http_host=WEAVIATE_HOST,
    http_port=WEAVIATE_PORT,
    http_secure=False,
    grpc_host="weaviate",
    grpc_port=50051,
    grpc_secure=False,
)
client = WeaviateClient(connection_params=connection_params)
client.connect()  # ✅ Required

embedding = OpenAIEmbeddings()

# === Vectorstore from existing index ===
vectorstore = WeaviateVectorStore(
    client=client,
    index_name="LangChain",  # Must match loader
    text_key="text",
    embedding=embedding,
)
retriever = vectorstore.as_retriever()


# === Tool wrapper ===
@tool
def search_docs(query: str) -> str:
    """Searches the knowledge base for the most relevant content."""
    # Assume retriever is already created
    docs = retriever.invoke(query)
    dcs_str = "\n\n".join([doc.page_content for doc in docs[:3]])
    print(f"docs retrieved: {dcs_str}")
    return dcs_str


llm = ChatOpenAI(temperature=0)
llm_with_tools = llm.bind_tools([search_docs])

retriever_tool = Tool(
    name="knowledge_base_search",
    func=search_docs,
    description=(
        "Always use this tool to answer any questions related to Estonian law, legal processes, or terms "
        "from the company documentation. For example, terms like 'advokatuur' or how to become a lawyer."
    )
)

memory = ConversationBufferMemory(return_messages=True)
# === User sessions with memory ===
user_sessions = {}


class ChatRequest(BaseModel):
    user_id: str
    message: str

system_message = (
    "You are a helpful assistant. "
    "Always use the knowledge_base_search tool if user asks about documentation, product features, instructions, or procedures."
    "The user and you are having a conversation. "
    "Remember and use the chat history (stored in `chat_history`) to answer follow-up questions. "
    "If the user has mentioned their name before, remember it."
)


@app.post("/chat")
async def chat(req: ChatRequest):
    user_id = req.user_id
    message = req.message

    if user_id not in user_sessions:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        user_agent = initialize_agent(
            tools=[retriever_tool],
            llm=ChatOpenAI(temperature=0.7),
            agent=AgentType.OPENAI_FUNCTIONS,
            agent_kwargs={"system_message": system_message},
            memory=memory,
            verbose=True
        )
        user_sessions[user_id] = user_agent

    agent = user_sessions[user_id]
    # DEBUG: Print chat history before agent response
    print("🔁 Memory BEFORE:")
    for msg in agent.memory.chat_memory.messages:
        print(msg)

    response = agent.run({"input": message})

    # DEBUG: Print chat history after agent response
    print("🧠 Memory AFTER:")
    for msg in agent.memory.chat_memory.messages:
        print(msg)
    return {"response": response}


@app.get("/", response_class=HTMLResponse)
async def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()
