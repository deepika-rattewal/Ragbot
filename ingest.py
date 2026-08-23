import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Configuration Constants (Engineering best practice: no magic numbers)
PDF_PATH = "data/sample.pdf"
DB_DIRECTORY = "chroma_db"

def run_pipeline():
    # Ensure PDF exists before starting
    if not os.path.exists(PDF_PATH):
        print(f"Error: Could not find {PDF_PATH}. Please put a PDF in the 'data/' folder.")
        return

    print("[1/4] Loading PDF document...")
    loader = PyPDFLoader(PDF_PATH)
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} pages from source document.")

    print("[2/4] Splitting text into manageable chunks...")
    # Chunk size 500 with overlap 50 ensures sentences don't get awkwardly cut off
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    docs = text_splitter.split_documents(raw_documents)
    print(f"Generated {len(docs)} text chunks.")

    print("[3/4] Initializing local open-source embedding model...")
    # We use a standard lightweight embedding model running locally via sentence-transformers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[4/4] Embedding text chunks and persisting to ChromaDB...")
    # This writes the vector database straight to your disk so it saves memory
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=DB_DIRECTORY
    )
    
    print(f"SUCCESS: Pipeline complete. Vector database stored locally at './{DB_DIRECTORY}'")

if __name__ == "__main__":
    run_pipeline()