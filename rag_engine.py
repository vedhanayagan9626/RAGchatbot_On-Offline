import os
from typing import Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline

class RAGEngine:
    def __init__(self):
        # Online RAG components
        self.vector_store = None
        self.qa_chain = None
        
        # Offline model components (Using a small pre-trained model for quick local inference)
        # We load this lazily to save startup time
        self.offline_pipeline = None

    def initialize_online_rag(self, pdf_path: str):
        """
        Reads the given PDF, chunks it, and builds a FAISS vector index using OpenAI embeddings.
        Sets up the RetrivalQA chain.
        """
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is missing.")

        print(f"Loading PDF: {pdf_path}")
        # Extract text from PDF
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
        
        # Split text into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        
        # Create embeddings and vector store
        print("Creating embeddings and FAISS index...")
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(texts, embeddings)
        
        # Initialize QA chain
        print("Initializing QA Chain...")
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) # Using a modern/efficient GPT model
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever()
        )
        print("Online RAG Engine initialized successfully.")

    def query_online_rag(self, query: str) -> str:
        """
        Queries the OpenAI GPT-based RAG application.
        """
        if not self.qa_chain:
            raise ValueError("Online RAG Engine is not initialized. Call /upload-pdf first or ensure the app loads the PDF on startup.")
        
        print(f"Processing Online LLM Query: {query}")
        result = self.qa_chain.invoke(query)
        return result["result"]

    def get_mock_api_data(self) -> Dict[str, Any]:
        """
        Simulates the hardcoded mock API response for server 10.200.2.192
        as requested in the assignment instructions.
        """
        return {
            "server": "10.200.2.192",
            "metrics": {
                "free_space": "150 GB",
                "cpu_utilization": "45%",
                "memory_usage": "16 GB"
            }
        }

    def query_offline_model(self, query: str) -> str:
        """
        Reads data from the mock API and uses a local offline HuggingFace model 
        to answer the user's question based on that API data.
        """
        # Load local HuggingFace offline model (e.g. TinyLlama or Flan-T5)
        # We use a smaller model here to ensure it runs without requiring huge GPU/RAM.
        if self.offline_pipeline is None:
            print("Loading offline Huggingface model... This might take a bit on the first run.")
            # google/flan-t5-small is a very efficient and small instruction-following model (~300MB)
            self.offline_pipeline = pipeline("text2text-generation", model="google/flan-t5-small")
        
        # 1. Read data from API
        print("Fetching data from Mock API...")
        api_data = self.get_mock_api_data()
        
        # 2. Formulate Prompt containing the API context
        context = (
            f"Here is the server data from the API: \n"
            f"Server IP: {api_data['server']}\n"
            f"Free Space: {api_data['metrics']['free_space']}\n"
            f"CPU Utilization: {api_data['metrics']['cpu_utilization']}\n"
            f"Memory Usage: {api_data['metrics']['memory_usage']}\n"
        )
        
        prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
        print(f"Processing Offline LLM Query: {query}")
        
        # 3. Generate Answer
        result = self.offline_pipeline(prompt, max_length=100)
        return result[0]["generated_text"]

# Singleton instance to be used by the router
rag_engine = RAGEngine()
