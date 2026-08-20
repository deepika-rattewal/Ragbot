import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Page configuration
st.set_page_config(
    page_title="Enterprise Doc Q&A Bot",
    page_icon="🤖",
    layout="centered"
)

DB_DIRECTORY = "chroma_db"
MODEL_NAME = "llama3"

@st.cache_resource
def load_qa_system():
    """
    Caches the QA system so it doesn't reload the models on every user click.
    This demonstrates software optimization knowledge.
    """
    if not os.path.exists(DB_DIRECTORY):
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=embeddings
    )
    llm = Ollama(model=MODEL_NAME)

    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you do not know the answer based on the context, state clearly "I cannot find the answer in the provided document." Do not hallucinate.

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# UI Layout Header
st.title("📄 Enterprise Document Q&A Assistant")
st.markdown("Query your technical manuals, research papers, or documentation securely with zero hallucinations.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load system
qa_system = load_qa_system()

if not qa_system:
    st.warning("⚠️ Vector database not found! Please run your `ingest.py` script first with a PDF file in the `data/` folder.")
else:
    # Accept user input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context..."):
                response_data = qa_system({"query": prompt})
                answer = response_data["result"]
                sources = response_data["source_documents"]

                # Format answer and sources
                full_response = f"{answer}\n\n**Sources:**"
                for doc in sources:
                    page_num = doc.metadata.get('page', 'Unknown')
                    full_response += f"\n- Page {page_num}"

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})