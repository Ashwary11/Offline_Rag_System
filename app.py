import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader, PyPDFLoader
import os

# Title
st.set_page_config(page_title="RAG System", layout="wide")
st.title("Minimal RAG System using HuggingFace & ChromaDB")

# Create folders if they don't exist
UPLOAD_DIR = "uploads"
CHROMA_DIR = "db"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Upload section
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Handle uploaded file
if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"Uploaded: {uploaded_file.name}")

    # Load document
    st.info("Reading and splitting document...")
    loader = PyPDFLoader(file_path) if uploaded_file.name.endswith(".pdf") else TextLoader(file_path)
    documents = loader.load()

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    st.success(f"Document split into {len(chunks)} chunks.")

    # Embedding and storing in Chroma
    st.info("Creating vector embeddings and storing in ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    st.success("Embedding stored successfully.")

# Query Section
query = st.text_input("Ask a question based on the document:")
if query:
    st.write("Searching for relevant chunks...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    results = vectordb.similarity_search(query, k=5)

    st.subheader("Top Matching Chunks:")
    for idx, doc in enumerate(results):
        st.markdown(f"**Chunk {idx+1}:**")
        st.write(doc.page_content)
        st.markdown("---")

    st.info("You can now use these chunks with any LLM like GPT/Gemini to generate answers.")
