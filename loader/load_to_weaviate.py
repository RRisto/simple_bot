import time
import os
import requests
from dotenv import load_dotenv

from weaviate.connect import ConnectionParams
from weaviate.client import WeaviateClient

from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load environment variables
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
DOCS_PATH = os.getenv("RAG_DATA_FOLDER", "rag_docs")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # Optional, for cloud setups


# Wait for Weaviate to become ready
def wait_for_weaviate(url, timeout=20):
    print(f"⏳ Waiting for Weaviate at {url}...")
    for _ in range(timeout):
        try:
            if requests.get(f"{url}/v1/.well-known/ready").status_code == 200:
                print("✅ Weaviate is ready!")
                return
        except:
            pass
        time.sleep(1)
    raise Exception("Weaviate didn't start in time")


wait_for_weaviate(WEAVIATE_URL)

# Load documents
loader = DirectoryLoader(
    path=DOCS_PATH,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
chunks = [doc for doc in chunks if doc.page_content.strip()]

# Create embedding model
embedding = OpenAIEmbeddings()

connection_params = ConnectionParams.from_params(
    http_host="weaviate",
    http_port=8080,
    http_secure=False,
    grpc_host="weaviate",
    grpc_port=50051,
    grpc_secure=False,
)

client = WeaviateClient(connection_params=connection_params)
client.connect()
# Upload to Weaviate
vectorstore = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    client=client,
    index_name="LangchainDocs",  # You can choose any name here
    text_key="text"
)

print(f"✅ Loaded {len(chunks)} chunks into Weaviate.")
