# RAG-PDF-QA

A lightweight Retrieval-Augmented Generation (RAG) system that lets you load a PDF manual and ask natural‑language questions about it.

The system extracts text from a PDF, splits it into chunks, stores semantic embeddings in a FAISS vector database, retrieves the most relevant passages, and uses a local LLM (via Ollama) to generate answers.

---

## Features

* Load and process PDF manuals
* Semantic search using embeddings
* Vector similarity retrieval with FAISS
* Local LLM answer generation using Ollama
* Interactive CLI for asking questions

---

## Architecture

```
PDF Document
      ↓
Text Extraction (PyPDF2)
      ↓
Chunking
      ↓
Embeddings (SentenceTransformers)
      ↓
FAISS Vector Database
      ↓
Retrieve Relevant Chunks
      ↓
Prompt Augmentation
      ↓
Local LLM Generation (Ollama)
```

---

## Installation

Clone the repository:

```
git clone https://github.com/AchintyaCh/RAG-PDF-QA.git
cd RAG-PDF-QA
```

Install dependencies:

```
pip install -r requirements.txt
```

Make sure Ollama is installed and a model is available (example):

```
ollama run granite3-moe:1b
```

---

## Usage

Run the application:

```
python app.py
```

Provide the path to a PDF manual when prompted and ask questions about it.

Example:

```
Enter Path of manual: washing_machine_manual.pdf
Question: How do I install the washing machine?
```

Type `exit` or `quit` to stop the program.

---

## Project Structure

```
RAG-PDF-QA
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Goal

This project demonstrates the core components of a RAG pipeline:

* document ingestion
* semantic retrieval
* vector search
* prompt augmentation
* local LLM generation

It serves as a simple prototype for building document‑based question answering systems.
