import os
import time
import requests
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from collections import defaultdict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from weaviate.connect import ConnectionParams
from weaviate.client import WeaviateClient
from weaviate.classes.config import Configure, Property, DataType

# Load env vars
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://weaviate:8080")
DOCS_PATH = os.getenv("RAG_DATA_FOLDER", "rag_docs")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")  # Optional for Weaviate Cloud


# --- Wait for Weaviate to become ready ---
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


# --- Load PDFs using PyMuPDF ---
def load_pdfs_with_metadata(folder_path):
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                filepath = os.path.join(root, file)
                try:
                    with pymupdf.open(filepath) as doc:
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        if text.strip():
                            documents.append(Document(page_content=text, metadata={"filename": file}))
                except Exception as e:
                    print(f"⚠️ Error reading {filepath}: {e}")
    return documents


print(f"📂 Loading PDFs from: {DOCS_PATH}")
docs = load_pdfs_with_metadata(DOCS_PATH)
print(f"📄 Loaded {len(docs)} PDF documents.")

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
chunks = [doc for doc in chunks if doc.page_content.strip()]
# Assign index per document (grouped by filename)
chunk_counters = defaultdict(int)

for chunk in chunks:
    filename = chunk.metadata.get("filename", "unknown")
    chunk.metadata["chunk_index"] = chunk_counters[filename]
    chunk_counters[filename] += 1
print(f"🧩 Split into {len(chunks)} text chunks.")

# --- Create embedding model ---
embedding = OpenAIEmbeddings()

# --- Connect to Weaviate ---
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

client.collections.create(
    name="LangchainDocs",
    vectorizer_config=Configure.Vectorizer.none(),
    properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="filename", data_type=DataType.TEXT),
        Property(name="chunk_index", data_type=DataType.INT),
        Property(name="page_number", data_type=DataType.INT),
    ]
)

# --- Upload to Weaviate ---
vectorstore = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=embedding,
    client=client,
    index_name="LangchainDocs",
    text_key="text"
)

client.close()

print(f"✅ Uploaded {len(chunks)} chunks to Weaviate.")
