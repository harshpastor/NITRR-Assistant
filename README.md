<<<<<<< HEAD

# 🎓 College Ordinance RAG Chatbot

An intelligent, context-aware AI assistant designed to answer student queries regarding college ordinances (B.Tech, B.Arch, etc.). It uses **Retrieval-Augmented Generation (RAG)** to provide accurate answers cited directly from official PDF documents.

## 🚀 Features

  * **Multi-Stream Support:** Separate knowledge bases for different programs (e.g., B.Tech, B.Arch).
  * **Powered by Google Gemini:** Uses `gemini-2.5-lite` for reasoning and `text-embedding-004` for vector search.
  * **Context-Aware:** Remembers chat history to handle follow-up questions (e.g., "How many semesters?" -\> "Total").
  * **Smart Ingestion:** Supports manual PDF uploads and bulk ingestion from a local folder.
  * **Interactive Citations:** Responses include citations that link directly to the source PDF.
  * **Markdown Support:** Beautifully formatted answers with bullet points, bold text, and lists.

## 🛠️ Tech Stack

**Backend:**

  * Python 3.12
  * Flask (API Server)
  * LangChain (Orchestration)
  * ChromaDB (Vector Database)
  * Google Generative AI SDK

**Frontend:**

  * React (Vite)
  * Tailwind CSS
  * React Markdown

-----

## ⚙️ Prerequisites

1.  **Python 3.11 or 3.12 (64-bit)** - *Critical: Do not use Python 3.13 or 3.14 yet.*
2.  **Node.js & npm** (for the frontend).
3.  **Google AI Studio API Key** (Get it [here](https://aistudio.google.com/)).

-----

## 📦 Installation Guide

### 1\. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration:**
Create a `.env` file in the `backend` folder:

```ini
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite
GEMINI_EMBED_MODEL=models/text-embedding-004
CHROMA_DIR=./chroma
DOCS_DIR=./docs
```

**Run Server:**

```bash
python app.py
```

### 2\. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
npm install react-markdown

# Run development server
npm run dev
```

-----

## 🐛 Challenges Faced & Solutions

During the development of this project, we encountered several specific environment and library issues. Here is how we resolved them:

### 1\. The Python 3.14 / 32-bit Compatibility Issue

  * **Error:** `Building wheel for numpy failed` or `ImportError: DLL load failed` when installing `chromadb`.
  * **Cause:** We initially tried using **Python 3.14 (Alpha)**. Many AI libraries (NumPy \< 2.0, ChromaDB) do not yet have pre-built binaries ("wheels") for Python 3.14 or 32-bit architectures, causing the system to try (and fail) to compile C++ code from source.
  * **Solution:** We uninstalled Python 3.14 and installed **Python 3.12 (64-bit)**. This allowed `pip` to download compatible pre-built wheels instantly.

### 2\. Google Gemini Model 404 Error

  * **Error:** `404 models/gemini-1.5-flash is not found for API version v1beta`.
  * **Cause:** The model alias `gemini-1.5-flash` was not resolving correctly for the specific API key tier/region, or the client library defaulted to an older version.
  * **Solution:** We ran a script to list available models and switched to the explicit versioned name: `gemini-1.5-flash-001`.

### 3\. ChromaDB Metadata Crash

  * **Error:** `ValueError: Expected metadata value to be a str, int, float or bool, got None`.
  * **Cause:** When ingesting PDFs, the `section` field was set to `None` by default. ChromaDB strictly forbids `None` in metadata.
  * **Solution:** Updated the ingestion pipeline to sanitize inputs, converting `None` values to empty strings `""` before insertion.

### 4\. Chat Context Memory

  * **Issue:** The bot treated every question as isolated. Asking "How many credits?" followed by "Total?" resulted in a search for the word "Total" (irrelevant results).
  * **Solution:** Implemented **Conversation History** passing from Frontend to Backend. The backend now performs "Query Expansion," combining the previous user message with the current one (e.g., "How many credits? Total") for the vector search.

-----

## 📂 Project Structure

```
/project-root
  ├── /backend
  │     ├── app.py                 # Main Flask Server
  │     ├── ingestion_pipeline.py  # CLI tool for batch ingestion
  │     ├── requirements.txt
  │     ├── .env
  │     ├── /docs                  # PDF storage
  │     └── /chroma                # Vector Database storage
  │
  └── /frontend
        ├── index.html
        ├── src/
        │    ├── App.jsx           # Main React Component
        │    ├── main.jsx
        │    └── ...
        └── package.json
```

## 📝 Usage

1.  Open the frontend (usually `http://localhost:5173`).
2.  Select a program (e.g., **B.Tech**).
3.  **Upload a PDF** using the side panel (or ensure files are in `backend/docs` and hit the endpoint).
4.  Ask questions\! Use the citations to view the source PDF.
=======
# College RAG Chatbot (OpenAI version)

This bundle swaps **Ollama** for the **OpenAI API** for generation and embeddings.

## Structure
```
college-rag-openai/
  backend/   # Flask + ChromaDB + OpenAI
  frontend/  # Vite + React (Tailwind via CDN)
```

## 1) Backend
See `backend/README.md`, or quick run:
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
.\venv\Scripts\Activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration:**
Create a `.env` file in the `backend` folder:

```ini
GOOGLE_API_KEY=your_actual_api_key_here
GEMINI_CHAT_MODEL=gemini-1.5-flash-001
GEMINI_EMBED_MODEL=models/text-embedding-004
CHROMA_DIR=./chroma
DOCS_DIR=./docs
```

**Run Server:**

```bash
python app.py
```

### 2\. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend
npm i
npm run dev  # open the URL printed by Vite
```
Update the **Backend URL** field in the header if your backend URL differs.

## Notes
- ChromaDB persists under `backend/chroma/` by default.
- The UI preserves all **variable names** and request/response shapes used earlier.
- Confidence gating switches between **RAG** and **GENERIC** modes; tweak threshold in `/ask` as needed.
>>>>>>> 6fc4862 (all files)
