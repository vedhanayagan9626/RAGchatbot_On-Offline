import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from typing import Dict, Any
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

class RAGEngine:
    def __init__(self):
        # Online RAG components
        self.vector_store = None
        self.compressor_retriever = None
        
        # Offline model components (Using a small pre-trained model for quick local inference)
        # We load this lazily to save startup time
        self.offline_pipeline = None

    def initialize_online_rag(self, pdf_path: str):
        """
        Reads the given PDF, chunks it, and builds a FAISS vector index using HuggingFace embeddings.
        """
        # Ensure at least one online API key is present
        if not os.environ.get("GROQ_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            logger.warning("Neither GROQ_API_KEY nor OPENAI_API_KEY is set. Online RAG will not work.")

        logger.info(f"Loading PDF: {pdf_path}")
        logger.info("Extracting text from PDF pages...")
        # Extract text from PDF
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
        
        logger.info(f"Successfully extracted {len(raw_text)} characters from {len(reader.pages)} pages.")
        
        # Split text into chunks using Recursive Splitter for semantic chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        
        # Create embeddings and vector store
        logger.info("Creating embeddings and FAISS index from document chunks...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_texts(texts, embeddings)
        
        # Implement Token Optimization: Contextual Compression
        # This will filter out irrelevant sentences from the retrieved chunks
        embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.6)
        self.compressor_retriever = ContextualCompressionRetriever(
            base_compressor=embeddings_filter, 
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 5})
        )
        
        logger.info("Online RAG Vector Store & Compressor initialized successfully.")

    def query_online_rag(self, query: str, provider: str = "groq") -> str:
        """
        Queries the online RAG application using the chosen provider (groq or gpt).
        """
        if not self.compressor_retriever:
            raise ValueError("Online RAG vector store is not initialized. Call /upload-pdf first or ensure the app loads the PDF on startup.")
        
        logger.info(f"Processing Online LLM Query using {provider.upper()}: {query}")
        
        if provider == "groq":
            if not os.environ.get("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY is not set.")
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        elif provider == "gpt":
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is not set.")
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        else:
            raise ValueError(f"Invalid provider: {provider}. Choose 'groq' or 'gpt'.")

        prompt_template = """Use the following pieces of context to answer the user's question. 
Answer strictly and concisely with only the final answer. Do not include any extra formatting, logs, or surrounding text.
If you do not know the answer based on the context, answer exactly with "I don't know".

Context:
{context}

Question: {question}

Answer:"""
        
        qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        qa_chain = qa_prompt | llm | StrOutputParser()

        grader_prompt = PromptTemplate.from_template(
            "You are a strict grader. Does this Answer directly address the Question based on the Context? "
            "If the answer says 'I don't know' or is unrelated, respond exactly with 'NO'. Otherwise, respond exactly with 'YES'.\n"
            "Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        )
        grader_chain = grader_prompt | llm | StrOutputParser()
        
        rewriter_prompt = PromptTemplate.from_template(
            "Rewrite the following query to be more specific and optimized for vector search retrieval: {query}"
        )
        rewriter_chain = rewriter_prompt | llm | StrOutputParser()

        MAX_RETRIES = 3
        attempt = 1
        current_query = query
        
        while attempt <= MAX_RETRIES:
            logger.info(f"--- ATTEMPT {attempt} ---")
            logger.info(f"Retrieving context for: '{current_query}'")
            docs = self.compressor_retriever.invoke(current_query)
            context_str = "\n\n".join(d.page_content for d in docs)
            
            logger.info("Generating Answer via LCEL Chain...")
            answer = qa_chain.invoke({"context": context_str, "question": current_query}).strip()
            logger.info(f"Draft Answer: {answer}")
            
            # Self-Healing Check
            logger.info("Evaluating Answer (Self-Reflection)...")
            score = grader_chain.invoke({"context": context_str, "question": current_query, "answer": answer}).strip().upper()
            
            if "YES" in score:
                logger.info("Evaluation Passed! Returning answer.")
                logger.info(f"Successfully processed {provider.upper()} response: \n{answer}\n")
                return answer
            else:
                logger.warning("Evaluation Failed (Hallucination/No Data). Initiating Self-Healing...")
                if attempt < MAX_RETRIES:
                    current_query = rewriter_chain.invoke({"query": current_query}).strip()
                    logger.info(f"Rewrote query to: '{current_query}'")
                attempt += 1

        # Conditional Termination
        termination_message = "I do not have enough context in the uploaded documents to answer this accurately."
        logger.error("Max retries reached. Triggering Conditional Termination.")
        return termination_message

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
        # Load local Ollama offline model
        if self.offline_pipeline is None:
            logger.info("Connecting to local Ollama instance... using llama3.2")
            self.offline_pipeline = Ollama(model="llama3.2")
        
        # 1. Read data from API
        logger.info("Fetching data from Mock API...")
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
        logger.info(f"Processing Offline LLM Query: {query}")
        
        # 3. Generate Answer
        logger.info("Invoking offline pipeline...")
        result = self.offline_pipeline.invoke(prompt)
        logger.info(f"Successfully processed offline response: \n{result}\n")
        return result

# Singleton instance to be used by the router
rag_engine = RAGEngine()
