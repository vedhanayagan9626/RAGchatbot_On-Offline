# AI Developer Assignment - Advanced Agentic RAG Chatbot

This repository contains an advanced RAG-based Chatbot built with Python, FastAPI, and LangChain. It features **Self-Healing AI**, **Contextual Compression Token Optimization**, and a robust **LangChain Expression Language (LCEL)** architecture. It supports both an online model (Groq llama3-8b-instant / OpenAI gpt-4o-mini) and an offline model (Ollama llama3.2).

---

## Architecture Highlights
- **Token Optimization**: Uses `RecursiveCharacterTextSplitter` and an `EmbeddingsFilter` to only send the most relevant sentences to the LLM.
- **Agentic Evaluator**: A built-in loop grades the LLM's own answer to prevent hallucinations.
- **Self-Healing Retries**: Automatically rewrites the user query and retries if the initial retrieval fails.
- **Dynamic Endpoints**: API accepts dynamic document uploads to instantly rebuild the vector store without restarting the server.

---

## Step 1: Environment Setup

1. **Open your terminal** and navigate to your project directory.
2. **Activate the virtual environment** (Highly Recommended to prevent IDE module errors):
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On MacOS/Linux
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   Run the following command to install all required libraries (including `langchain-groq`, `langchain-openai`, and `python-multipart`).
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure API Keys:**
   Create a `.env` file in the root directory. Add your desired keys:
   ```env
   GROQ_API_KEY=your_groq_key_here
   OPENAI_API_KEY=your_openai_key_here
   ```
   *(Note: Only GROQ is required for default functionality, but you can add OPENAI if you want to test provider switching).*

---

## Step 2: Starting the Servers

1. **Start the FastAPI Server**:
   In your activated terminal, run:
   ```bash
   uvicorn main:app --reload
   ```
2. **Start the Ollama Server (For Offline Mode)**:
   Ensure your local Ollama desktop application is running in the background. If you haven't downloaded the offline model yet, run:
   ```bash
   ollama pull llama3.2
   ```

---

## Step 3: Step-by-Step Execution Guide

To test the application properly, follow these strict sequential steps using the provided **Postman Collection**.

1. **Import the Collection:**
   Open Postman, click **Import**, and select the `postman_collection.json` file located in this repository.

### Action 1: Upload the Data (Mandatory First Step)
Because the RAG engine is dynamic, it does not load data on startup. You must upload the PDF into the server memory first!
1. Select the **"Upload PDF Document"** request in Postman.
2. Go to the **Body** tab. Hover over the `file` value field and click "Select Files".
3. Choose the `Platforms Supported.pdf` file.
4. Hit **Send**. You will receive a success message, and the terminal will log the text extraction and FAISS indexing process.

### Action 2: Test the Online RAG
1. Select **"Online Chatbot"**.
2. Go to the **Body** tab and modify the `query` field to ask any question about the uploaded PDF.
3. Hit **Send**.
3. **Watch the Terminal logs!** You will see the Advanced Agentic Architecture in action:
   - Extracting context via `ContextualCompressionRetriever`.
   - Generating a draft answer via LCEL chain.
   - Performing a *Self-Reflection Evaluation* to check for hallucinations.
   - Outputting the verified answer in Postman!
4. *(Optional)* You can change `"provider": "groq"` to `"provider": "gpt"` in the JSON body to dynamically switch LLMs!

### Action 3: Test the Offline Chatbot
1. Select **"Offline Chatbot"**.
2. Go to the **Body** tab and modify the `query` field to ask about server mock data (e.g., Free Space or CPU).
3. Hit **Send**.
3. **Wait for Cold Start:** If this is your first time querying the offline model, Ollama will take 10-40 seconds to load the 2GB model into your computer's RAM. All subsequent requests will be instant!
4. The API will respond natively without using the internet, generating an answer based on the mock server data injected into the prompt.

---

## Deliverables Included
- `main.py` & `rag_engine.py`: Fully modular backend code.
- `requirements.txt`: Locked dependencies.
- `postman_collection.json`: Testing endpoints.
- `README.md`: This comprehensive step-by-step guide.
