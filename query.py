import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration constants matching our ingestion script
DB_DIRECTORY = "chroma_db"

def search_documents(query_text: str, k: int = 3):
    """
    Searches the local Chroma vector database for the top-k most 
    semantically similar text chunks related to the query.
    """
    if not os.path.exists(DB_DIRECTORY):
        print(f"Error: Database directory '{DB_DIRECTORY}' not found. Run ingest.py first.")
        return []

    print("[1/2] Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[2/2] Connecting to ChromaDB and executing similarity search...")
    # Load the existing database from disk instead of re-processing
    vector_store = Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings
    )

    # Perform similarity search with score (lower score means higher similarity in L2 space)
    results = vector_store.similarity_search_with_score(query_text, k=k)
    
    return results

if __name__ == "__main__":
    # Test query to verify retrieval logic works standalone
    test_query = "What is this document about?"
    print(f"\nExecuting test search for: '{test_query}'\n")
    
    matched_docs = search_documents(test_query, k=2)

    for i, (doc, score) in enumerate(matched_docs, 1):
        print(f"--- Result {i} (Distance Score: {score:.4f}) ---")
        print(doc.page_content.strip())
        print(f"Source Page: {doc.metadata.get('page', 'Unknown')}\n")
        