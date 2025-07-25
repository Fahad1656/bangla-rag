# api/routes/inference.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rag_pipeline.rag import MiniRAG
from models.model import create_embedder, TextChunkEmbedder
from storage.store import create_vector_store, build_vector_store_from_embeddings
from data_preprocessing.extract import extract_pdf_content
from data_preprocessing.preprocess_data import preprocess_document
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Env settings
PDF_FILE_PATH = os.getenv("PDF_FILE_PATH")
VECTOR_STORE_FOLDER = os.getenv("VECTOR_STORE_FOLDER")
CACHE_FOLDER = os.getenv("CACHE_FOLDER")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
RAG_MODEL_NAME = os.getenv("RAG_MODEL_NAME")
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))

# Load embedder and vector store
embedding_cache_dir = os.path.join(CACHE_FOLDER, "embeddings")
embedder = create_embedder(model_name=EMBEDDING_MODEL_NAME, cache_dir=embedding_cache_dir)

faiss_index_path = os.path.join(VECTOR_STORE_FOLDER, "faiss_index.idx")
if os.path.exists(faiss_index_path):
    print("📦 Loading existing vector store from disk...")
    vector_store = create_vector_store(embedder.get_embedding_dimension(), store_dir=VECTOR_STORE_FOLDER)
else:
    print("📄 Extracting content from PDF and creating vector store...")
    document_content = extract_pdf_content(PDF_FILE_PATH)
    text_chunks = preprocess_document(document_content, chunk_size=CHUNK_SIZE)
    chunk_embedder = TextChunkEmbedder(embedder)
    embedded_chunks = chunk_embedder.embed_chunks(text_chunks)
    vector_store = build_vector_store_from_embeddings(embedded_chunks, store_dir=VECTOR_STORE_FOLDER)
    vector_store.save_index()

# Initialize RAG pipeline
rag_system = MiniRAG(vector_store=vector_store,
                     embedder=embedder,
                     model=RAG_MODEL_NAME,
                     base_url=OLLAMA_API_URL)

# Request body
class QuestionRequest(BaseModel):
    question: str

# POST endpoint
@router.post("/ask")
async def ask_question(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        result = rag_system.get_answer(question)
        if result.language=="bn":
            result.language = "বাংলা" 
        if result.language=="en":
            result.language = "English"
        return {
            "answer": result.answer,
            "language prompted": result.language,
            "confidence": result.confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during processing: {str(e)}")

