import os
import shutil
import tempfile
import streamlit as st
import pytesseract

from pdf2image import convert_from_path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    TextLoader,
)

# ---------------------------------------------------
# STREAMLIT CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Ashwary Offline Retrieval-Augmented Generation (RAG) System",
    layout="wide"
)

st.title("Ashwary Offline RAG System ")
st.write("Supports **multiple files**, **scanned PDFs**, up to **500 MB**.")

# Increase upload limit (Streamlit safe limit)
st.config.set_option("server.maxUploadSize", 500)

# ---------------------------------------------------
# OCR FUNCTION
# ---------------------------------------------------
def ocr_pdf_to_documents(pdf_path):
    images = convert_from_path(pdf_path)
    full_text = ""

    for img in images:
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"

    if not full_text.strip():
        return []

    return [Document(page_content=full_text)]

# ---------------------------------------------------
# FILE UPLOAD (MULTIPLE)
# ---------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF or TXT files (Max 500MB each)",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

# ---------------------------------------------------
# TEMP DIRECTORY
# ---------------------------------------------------
temp_dir = tempfile.mkdtemp()
documents = []

# ---------------------------------------------------
# LOAD FILES
# ---------------------------------------------------
for uploaded_file in uploaded_files:
    file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info(f"Processing: {uploaded_file.name}")

    # PDF handling
    if uploaded_file.name.lower().endswith(".pdf"):
        loader = UnstructuredPDFLoader(file_path)
        docs = loader.load()

        # OCR fallback
        if not docs or not docs[0].page_content.strip():
            st.warning(f"OCR triggered for scanned PDF: {uploaded_file.name}")
            docs = ocr_pdf_to_documents(file_path)

        documents.extend(docs)

    # TXT handling
    elif uploaded_file.name.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        documents.extend(docs)

# ---------------------------------------------------
# VALIDATION
# ---------------------------------------------------
if not documents:
    st.error("No readable text found in uploaded files.")
    shutil.rmtree(temp_dir)
    st.stop()

st.success(f"Loaded {len(documents)} document(s)")

# ---------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

st.success(f"Generated {len(chunks)} text chunks")

# ---------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------
# VECTOR STORE
# ---------------------------------------------------
persist_dir = os.path.join(temp_dir, "chroma_db")
db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=persist_dir
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# ---------------------------------------------------
# QUERY INPUT
# ---------------------------------------------------
query = st.text_input("Ask a question from the uploaded documents")

if query:
    with st.spinner("Retrieving relevant information..."):
        relevant_docs = retriever.get_relevant_documents(query)

    st.subheader("Retrieved Context")
    for i, doc in enumerate(relevant_docs, 1):
        st.markdown(f"**Chunk {i}**")
        st.code(doc.page_content[:800])
        st.markdown("---")

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
Use the following context to answer the question accurately.
If the answer is not present, say "Information not found".

Context:
{context}

Question:
{query}

Answer:
"""

    st.subheader("Prompt Sent to LLM")
    st.code(prompt)

    try:
        llm = Ollama(model="mistral")
        answer = llm.invoke(prompt)

        st.subheader("Final Answer")
        st.write(answer)

    except Exception as e:
        st.error(f"Ollama Error: {str(e)}")

# ---------------------------------------------------
# CLEANUP
# ---------------------------------------------------
st.divider()
if st.button("Clear Session"):
    shutil.rmtree(temp_dir, ignore_errors=True)
    st.experimental_rerun()
