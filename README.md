Offline RAG System with OCR Support (Local GenAI)

This project implements a fully local Retrieval-Augmented Generation (RAG) system that can process readable PDFs, scanned PDFs, and text files, and answer user queries using a locally hosted LLM.

The system is designed for privacy-first environments, where documents must remain on the local machine and cloud APIs are not allowed. All processing — OCR, embeddings, retrieval, and response generation — happens entirely offline.

✨ Key Features

🔒 Fully Local & Offline

No cloud APIs

No external data transfer

Runs entirely on localhost

📂 Multi-File Upload (Up to 500 MB per file)

Supports uploading multiple PDF and TXT files

Handles large documents efficiently

🧾 Readable + Scanned PDF Support

Extracts text from digital PDFs

Automatically falls back to OCR for scanned PDFs

🔤 OCR Integration

Uses Tesseract OCR to extract text from image-based PDFs

Ensures scanned documents are searchable and queryable

🧠 Retrieval-Augmented Generation (RAG)

Retrieves relevant document chunks before generating answers

Reduces hallucinations and improves answer grounding

🗂️ ChromaDB Vector Store

Stores document embeddings locally

Enables fast semantic similarity search

🤖 Local LLM via Ollama

Uses the Mistral model hosted locally

No internet required at runtime

🖥️ Interactive Streamlit UI

Upload documents

Ask questions

View retrieved context and final answers

🛠️ How the System Works
1️⃣ File Upload

Users upload one or more PDF or TXT files via the Streamlit interface.

Each file can be up to 500 MB.

2️⃣ Document Loading & OCR

Readable PDFs are processed using UnstructuredPDFLoader.

If no readable text is found:

OCR is automatically triggered using Tesseract

PDF pages are converted to images

Text is extracted via OCR

TXT files are loaded directly using TextLoader.

3️⃣ Text Chunking

Extracted text is split into overlapping chunks using:

RecursiveCharacterTextSplitter

Chunk size: 500 characters

Overlap: 100 characters

This improves retrieval accuracy.

4️⃣ Embedding Generation

Each text chunk is converted into vector embeddings using:

sentence-transformers/all-MiniLM-L6-v2

Embeddings are generated locally.

5️⃣ Vector Storage

All embeddings are stored in a local ChromaDB instance.

The database persists inside a temporary local directory.

6️⃣ Query & Retrieval

User enters a natural language question.

ChromaDB retrieves the top-K most relevant chunks based on semantic similarity.

Retrieved chunks are shown in the UI for transparency.

7️⃣ Answer Generation

Retrieved context is combined with the user’s query.

The augmented prompt is sent to Mistral via Ollama.

The model generates a grounded response.

If the answer is not found in the documents, the model is instructed to say so.

8️⃣ Session Control

Users can clear the session, which:

Deletes temporary files

Clears the vector store

Resets the app state

🔧 Tech Stack

Frontend: Streamlit

LLM: Mistral (served locally via Ollama)

Vector Database: ChromaDB

Embeddings: SentenceTransformers (MiniLM)

OCR: Tesseract OCR

PDF Processing: Unstructured, pdf2image

Runtime: Fully local / offline capable

🎯 Use Cases

Secure document Q&A

Internal knowledge assistants

Government or enterprise environments

Offline AI systems

Privacy-sensitive document analysis

Scanned document intelligence

🚀 Why This Project Matters

This project demonstrates that GenAI systems can be powerful without cloud dependency.
By combining local LLMs, OCR, and vector search, it enables secure, explainable, and production-ready RAG pipelines that respect data privacy and operational constraints.
