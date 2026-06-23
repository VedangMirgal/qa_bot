# RAG-Based Enterprise Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application that enables users to ask natural language questions about PDF documents and receive context-aware answers grounded in document content.

The application combines document retrieval with Google's Gemini models to improve response relevance and reduce hallucinations when working with document-based knowledge.

---

## Overview

Traditional large language models cannot reliably answer questions about information contained within private documents unless that information is included in the prompt.

This project addresses that challenge by implementing a Retrieval-Augmented Generation (RAG) pipeline:

1. Upload a PDF document.
2. Extract and preprocess document text.
3. Split content into manageable chunks.
4. Generate vector embeddings for semantic search.
5. Store embeddings in a vector database.
6. Retrieve relevant document sections for a user query.
7. Generate a response using Gemini based on retrieved context.

This approach allows answers to remain grounded in the uploaded document rather than relying solely on the model's pretrained knowledge.

---

## Features

* PDF document ingestion
* Automated document chunking
* Semantic search using vector embeddings
* Retrieval-Augmented Generation (RAG)
* Context-aware question answering
* Interactive Gradio interface
* Gemini-powered response generation

---

## System Architecture

```text
PDF Document
      │
      ▼
Document Loader
      │
      ▼
Text Chunking
      │
      ▼
Embedding Generation
      │
      ▼
ChromaDB Vector Store
      │
      ▼
Retriever
      │
      ▼
Gemini LLM
      │
      ▼
Answer Generation
```

---

## Technology Stack

### AI & Retrieval

* LangChain
* Google Gemini 2.5 Flash
* Hugging Face Embeddings

  * all-MiniLM-L6-v2
* ChromaDB

### Backend

* Python

### Interface

* Gradio

### Document Processing

* PyPDFLoader

---

## Project Structure

```text
.
├── qabot.py
├── .env
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

Windows:

```bash
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Running the Application

```bash
python qabot.py
```

The Gradio interface will launch locally and open in your browser.

---

## Example Workflow

1. Upload a PDF document.
2. Enter a question such as:

```text
What are the key findings discussed in this document?
```

3. The system retrieves the most relevant document sections.
4. Gemini generates an answer using the retrieved context.
5. The response is displayed through the Gradio interface.

---

## Learning Outcomes

This project provided hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Vector databases
* Semantic search
* Embedding models
* LLM integration
* Document intelligence systems
* LangChain pipelines

---

## Future Improvements

Potential extensions include:

* Multi-document support
* Persistent vector storage
* Citation-aware responses
* Conversation memory
* Hybrid search (keyword + semantic retrieval)
* Support for DOCX and TXT documents

---

## License

This project is intended for educational and learning purposes.
