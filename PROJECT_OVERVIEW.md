# Advanced Agentic RAG Chatbot: Project Overview

This document provides a comprehensive overview of the Retrieval-Augmented Generation (RAG) system built in this repository, detailing its architecture, competitive advantages, and a quick-start guide for other developers to utilize the code.

---

## 🧠 System Architecture & Tech Stack

This project is built using a modern, highly modular tech stack designed for speed, flexibility, and intelligent reasoning.

### Backend Infrastructure
- **FastAPI**: Serves as the core web framework, handling concurrent requests, API routing, and serving the static frontend files.
- **LangChain (LCEL)**: The backbone of the RAG engine. Instead of legacy chains, we utilize the modern LangChain Expression Language (LCEL) for dynamic prompt chaining and output parsing.
- **FAISS (Facebook AI Similarity Search)**: An in-memory vector database used to store and rapidly retrieve semantic embeddings.

### Intelligence Layers (LLMs)
The system features a **Dual-Mode Intelligence** architecture, allowing seamless switching between models based on security, speed, or internet availability:
1. **Online Models (High Speed / Cloud)**:
   - **Groq**: Utilizing `llama-3.1-8b-instant` via LPUs for lightning-fast inference.
   - **OpenAI**: Utilizing `gpt-4o-mini` for highly accurate, standardized cloud generation.
2. **Offline Model (Privacy / Local)**:
   - **Ollama**: Utilizing the local `llama3.2` model. This runs 100% natively on the host machine, requiring absolutely no internet connection, ensuring data privacy for sensitive mock API data.

### Frontend UI
- **Vanilla HTML/CSS/JS**: A sleek, dependency-free web interface featuring a modern "glassmorphism" aesthetic, dynamic loading animations, and seamless asynchronous `fetch` integration with the backend endpoints.

---

## 🚀 Key Advantages & Innovations

What sets this RAG system apart from standard implementations?

1. **Contextual Compression (Token Optimization)**
   Instead of feeding massive 1000-character chunks directly to the LLM (which wastes tokens and causes hallucination), we use an `EmbeddingsFilter`. This acts as a semantic highlighter—it extracts and passes *only the exact sentences* relevant to the user's query.
2. **Agentic "Self-Healing" Loop**
   The LLM does not just blindly answer. It utilizes a **Critic/Evaluator Pattern**:
   - The LLM drafts an answer.
   - A Grader Prompt forces the LLM to verify if its own answer is strictly grounded in the PDF.
   - If it detects a hallucination, it *rewrites the user's query* to be more optimal for vector search and retries.
3. **Dynamic Knowledge Injection**
   The vector store is not hardcoded. The application exposes an `/upload-pdf` endpoint (accessible via the UI or Postman) that dynamically extracts text, chunks it recursively, and instantly rebuilds the active FAISS database on the fly.
4. **Conditional Termination**
   If the Self-Healing loop fails 3 times, the system refuses to hallucinate. It triggers a graceful conditional termination, reliably returning an "I do not have enough context" message.

---

## 🛠 How Others Can Utilize This Code

This repository is designed to be plug-and-play. Other developers can easily clone and adapt this system for their own custom Chatbots.

### Step 1: Clone the Repository
```bash
git clone <your-repository-url>
cd RAGchatbot_On-Offline
```

### Step 2: Setup the Virtual Environment
To avoid package conflicts, it is highly recommended to use a virtual environment.
```bash
python -m venv venv

# Activate on Windows:
.\venv\Scripts\activate

# Activate on MacOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
The `requirements.txt` file is locked with exact versions (including `python-multipart` and `langchain-groq`) for guaranteed stability.
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a file named `.env` in the root directory. You must supply your own API keys for the cloud models.
```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Step 5: Start Local Services
1. **Start Ollama** (Required only if testing the Offline model):
   Ensure your local Ollama desktop app is running, and pull the required model in a separate terminal:
   ```bash
   ollama pull llama3.2
   ```
2. **Start the FastAPI Server**:
   ```bash
   uvicorn main:app --reload
   ```

### Step 6: Use the Web Interface!
Simply open a web browser and navigate to:
**http://127.0.0.1:8000**

1. Click the **Upload** icon in the top right to upload any custom PDF.
2. Select your desired model (Groq, OpenAI, or Local Ollama) from the dropdown.
3. Start chatting with your data!
