import os
import shutil
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader, TextLoader
import tempfile

st.set_page_config(page_title="RAG with Mistral", layout="wide")
st.title("RAG App")

# Upload PDF/TXT file
uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

if uploaded_file is not None:
    # Save file temporarily
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Load document (with OCR support)
    if uploaded_file.name.endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    documents = loader.load()

    if not documents:
        st.error("No readable text found in the document.")
        st.stop()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        st.error("Failed to split document into chunks.")
        st.stop()

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create a new temporary vector store directory
    persist_dir = os.path.join(temp_dir, "chroma_db")
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # Clear previous vector store if any

    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=persist_dir)

    # Query input
    query = st.text_input("Ask a question from the document")

    if query:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.get_relevant_documents(query)

        # Show chunk previews
        st.subheader("Retrieved Chunks (Top 3):")
        for i, doc in enumerate(relevant_docs):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(doc.page_content[:500])
            st.markdown("---")

        # Build prompt
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        prompt = f"""Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""

        st.subheader("Prompt Sent to Mistral:")
        st.code(prompt)

        # Get answer from Mistral via Ollama
        try:
            llm = Ollama(model="mistral")
            final_answer = llm.invoke(prompt)
            st.subheader("Final Answer from Mistral:")
            st.write(final_answer)
        except Exception as e:
            st.error(f"Failed to get response from Mistral: {str(e)}")
