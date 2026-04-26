from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables (e.g. GROQ_API_KEY and OPENAI_API_KEY)
load_dotenv()

# Set HuggingFace cache directory to D: drive to avoid disk space issues
cache_dir = os.path.join('D:', 'hf_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir

from rag_engine import rag_engine

app = FastAPI(
    title="AI Developer Chatbot APIs",
    description="RAG-based Chatbot endpoints for both Online (GPT-based) and Offline (HuggingFace) models.",
    version="1.0.0"
)

# Vector store is initialized via the /upload-pdf endpoint only.

class ChatRequest(BaseModel):
    query: str
    provider: str = "groq"  # "groq" or "gpt"

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat/online", response_model=ChatResponse)
def chat_online(request: ChatRequest):
    """
    Endpoint for Chatbot using Online model (Groq or GPT).
    Expects `provider` to be "groq" or "gpt". Defaults to "groq".
    """
    if request.provider == "groq" and not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in the environment.")
    if request.provider == "gpt" and not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in the environment.")
        
    if not rag_engine.vector_store:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet. Please use the /upload-pdf endpoint first.")

    try:
        answer = rag_engine.query_online_rag(request.query, provider=request.provider)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a new PDF document.
    Saves the file and rebuilds the FAISS vector index dynamically.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    temp_file_path = f"uploaded_{file.filename}"
    
    try:
        # Save uploaded file locally
        with open(temp_file_path, "wb") as buffer:
            buffer.write(await file.read())
            
        # Re-initialize the RAG index with the new file
        rag_engine.initialize_online_rag(temp_file_path)
        
        return {
            "message": "PDF uploaded and vector index rebuilt successfully.",
            "filename": file.filename
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/chat/offline", response_model=ChatResponse)
def chat_offline(request: ChatRequest):
    """
    Endpoint for Chatbot using Offline model (Ollama).
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
