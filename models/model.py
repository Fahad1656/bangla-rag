import os
from sentence_transformers import SentenceTransformer
import numpy as np
import hashlib
import pickle
from pathlib import Path
from loguru import logger
from typing import List, Dict, Any


CACHE_FOLDER = os.getenv("CACHE_FOLDER")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")



class MultilingualEmbedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, device=None, cache_dir=None):
        self.cache_dir = Path(cache_dir or CACHE_FOLDER + "/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
        logger.info(f"Loading embedder {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
        self.embedding_cache = {}
        self._load_cache()

    def _load_cache(self):
        cache_file = self.cache_dir / "embed_cache.pkl"
        if cache_file.exists():
            try:
                self.embedding_cache = pickle.load(open(cache_file, "rb"))
                logger.info(f"Loaded {len(self.embedding_cache)} embeddings from cache")
            except:
                self.embedding_cache = {}

    def save_cache(self):
        pickle.dump(self.embedding_cache, open(self.cache_dir / "embed_cache.pkl", "wb"))

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def preprocess(self, text: str, is_query=False):
        text = text.strip()
        text = " ".join(text.split())
        prefix = "query:" if is_query else "passage:"
        if not text.startswith(prefix):
            text = f"{prefix} {text}"
        return text

    def encode_batch(self, texts: List[str], batch_size=32, normalize=True, is_query=False):
        processed = [self.preprocess(t, is_query=is_query) for t in texts]
        embeddings = self.model.encode(processed, batch_size=batch_size, normalize_embeddings=normalize, show_progress_bar=False)
        return np.array(embeddings)

    def encode_query(self, query: str):
        return self.encode_batch([query], batch_size=1, is_query=True)[0]

    def get_embedding_dimension(self):
        return self.embedding_dimension

    
    def clear_cache(self):
        self.embedding_cache.clear()
        (self.cache_dir / "embed_cache.pkl").unlink(missing_ok=True)
        logger.info("Cleared embedding cache")


class TextChunkEmbedder:
    def __init__(self, embedder: MultilingualEmbedder):
        self.embedder = embedder

    def embed_chunks(self, chunks: List[Any], batch_size=32) -> Dict[str, Any]:
        texts = [c.content for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        metadata = [
            {
                "chunk_id": c.chunk_id,
                "source_page": c.source_page,
                "language": c.language,
                "content_type": c.content_type,
                "word_count": c.word_count,
                "char_count": c.char_count
            }
            for c in chunks
        ]

        logger.info(f"Embedding {len(texts)} chunks")
        embeddings = self.embedder.encode_batch(texts, batch_size=batch_size)

        result = {
            "embeddings": embeddings,
            "chunk_ids": chunk_ids,
            "texts": texts,
            "metadata": metadata,
            "embedding_dimension": self.embedder.embedding_dimension,
            "total_chunks": len(chunks)
        }
        return result


def create_embedder(model_name=EMBEDDING_MODEL_NAME, device=None, cache_dir=CACHE_FOLDER):
    return MultilingualEmbedder(model_name=model_name, device=device, cache_dir=cache_dir)


