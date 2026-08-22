import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Configuration constants
DB_DIRECTORY = "chroma_db" 
# If you prefer using an API instead of Ollama locally, you can swap this, 
# but 'llama3' running via Ollama is a great local systems demo.
MODEL_NAME = "llama3"

def build_qa_chain():
    """
    Initializes the local vector store, wraps it with a retrieval chain,
    and configures a strict prompt template to prevent hallucinations.
    """
    if not os.path.exists(DB_DIRECTORY):
        print(f"Error: Database directory '{DB_DIRECTORY}' not found. Run ingest.py first.")
        return None

    print("[1/3] Loading local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("[2/3] Connecting to vector database...")
    vector_store = Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings
    )

    print(f"[3/3] Initializing local LLM ({MODEL_NAME}) via Ollama...")
    # Connects to Ollama running locally on your machine
    llm = OllamaLLM(model=MODEL_NAME)

    # Engineering safeguard: Force the model to ONLY use the provided context
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you do not know the answer based on the context, state clearly "I cannot find the answer in the provided document." Do not try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Build the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain

if __name__ == "__main__":
    # Make sure Ollama is installed and you ran 'ollama run llama3' at least once in your terminal
    qa = build_qa_chain()
    
    if qa:
        test_question = "What is the primary topic of this document?"
        print(f"\nAsking question: '{test_question}'\n");
        
        response = qa({"query": test_question})
        
        print("--- AI Answer ---")
        print(response["result"].strip())
        print("\n--- Sources Used ---")
        for doc in response["source_documents"]:
            print(f"- Page {doc.metadata.get('page', 'Unknown')}");