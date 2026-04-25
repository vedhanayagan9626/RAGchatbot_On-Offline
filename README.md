# AI Developer Assignment - RAG Chatbot

This repository contains a RAG-based Chatbot built with Python, FastAPI, and LangChain, featuring both an online model (GPT-4o-mini) and an offline model (HuggingFace flan-t5-small).

## Prerequisites
- Python 3.9+ installed on your system.
- An OpenAI API Key.
- Postman Desktop Client.

---

## Step 1: Environment Setup

1. **Open your terminal or command prompt** and navigate to your project directory.
2. **(Optional but recommended)** Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On MacOS/Linux
   source venv/bin/activate
   ```
3. **Install Dependencies:**
   Run the following command to install all the required libraries explicitly listed with versions in the `requirements.txt` file.
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables:**
   Create a new file named `.env` in the same directory as the script. Add your OpenAI API key to it:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## Step 2: Running the Application

1. Ensure the provided sample PDF (`Platforms Supported.pdf`) is in the same directory as `main.py`.
2. Start the FastAPI server using `uvicorn`:
   ```bash
   python main.py
   ```
   *Alternatively:*
   ```bash
   uvicorn main:app --host 127.0.0.1 --port 8000 --reload
   ```
3. Look at the terminal console output! 
   You should see logs indicating:
   - `Loading PDF: Platforms Supported.pdf`
   - `Creating embeddings and FAISS index...`
   - `Application startup complete.`

---

## Step 3: Testing using Postman Collection

1. Open **Postman**.
2. Click on **Import** in the top left corner of the Postman application.
3. Select the `postman_collection.json` file generated in this directory and import it.
4. You will see a new collection named **"AI Developer API Collection"**. Expand it to view the pre-configured requests.

### Testing Online Model
1. Select **"1. Online Chatbot - WebLogic Versions"**.
2. Assuming your local server is running, hit **Send**.
3. View the response! The online GPT-based model will read the PDF FAISS index and return the answer about WebLogic versions supported.
   *Take your screenshot of this request and response as required by the assignment.*

### Testing Offline Model
1. Select **"3. Offline Chatbot - Free Space"**.
2. Hit **Send**.
3. *Note:* On the very first run, it will take a few seconds to download the lightweight `google/flan-t5-small` huggingface model to your system.
4. You will see clear **Console Logs** in your terminal indicating:
   - `Fetching data from Mock API...`
   - `Processing Offline LLM Query...`
5. The API will respond with the answer generated natively on your machine based on the mock data injected.
   *Take your screenshot of this request and response.*

---

## Deliverables Generated:
- `main.py` and `rag_engine.py`: Python scripts with clear comments and modular structure.
- `requirements.txt`: Project dependencies with library versions.
- `postman_collection.json`: The fully configured Postman collection.
