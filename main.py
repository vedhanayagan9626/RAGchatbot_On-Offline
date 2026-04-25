from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()

from rag_engine import rag_engine

app = FastAPI(
    title="AI Developer Chatbot APIs",
    description="RAG-based Chatbot endpoints for both Online (GPT-based) and Offline (HuggingFace) models.",
    version="1.0.0"
)

# Initialize the Online RAG model on startup if OPENAI_API_KEY is available
PDF_FILE_PATH = "Platforms Supported.pdf"

@app.on_event("startup")
async def startup_event():
    # Attempt to initialize the online RAG if API key is provided
    if os.environ.get("OPENAI_API_KEY"):
        try:
            rag_engine.initialize_online_rag(PDF_FILE_PATH)
        except Exception as e:
            print(f"Error initializing online RAG on startup: {e}")
    else:
        print("Warning: OPENAI_API_KEY not found in environment. Online RAG will not work.")

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat/online", response_model=ChatResponse)
async def chat_online(request: ChatRequest):
    """
    Endpoint for Chatbot using Online model (GPT-4o / GPT-based).
    Expects `OPENAI_API_KEY` in the `.env` file.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in the environment.")
    if not rag_engine.qa_chain:
        # Try initializing again
        try:
            rag_engine.initialize_online_rag(PDF_FILE_PATH)
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Failed to initialize Online RAG: {e}")

    try:
        answer = rag_engine.query_online_rag(request.query)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/offline", response_model=ChatResponse)
async def chat_offline(request: ChatRequest):
    """
    Endpoint for Chatbot using Offline model (HuggingFace Pipeline).
    Reads server data from a mock hardcoded API and answers the query.
    """
    try:
        answer = rag_engine.query_offline_model(request.query)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
