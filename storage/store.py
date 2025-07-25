# -*- coding: utf-8 -*-

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from loguru import logger
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    chunk_index: int


class FAISSVectorStore:
    def __init__(self, embedding_dimension: int, index_type="flat", metric="cosine",
                 store_dir: Optional[str] = None):
        self.embedding_dimension = embedding_dimension
        self.index_type = index_type
        self.metric = metric
        self.store_dir = Path(store_dir or "vector_store")
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index = self._create_index()
        self.chunk_ids, self.chunk_texts, self.chunk_metadata = [], [], []
        self.id_to_index, self.index_to_id = {}, {}
        logger.info(f"Initialized FAISS store dim={embedding_dimension}, type={index_type}, metric={metric}")

    def _create_index(self):
        if self.metric == "cosine":
            if self.index_type == "flat":
                return faiss.IndexFlatIP(self.embedding_dimension)
            else:
                quant = faiss.IndexFlatIP(self.embedding_dimension)
                return faiss.IndexIVFFlat(quant, self.embedding_dimension, 100)
        elif self.metric == "l2":
            if self.index_type == "flat":
                return faiss.IndexFlatL2(self.embedding_dimension)
            quant = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.embedding_dimension), self.embedding_dimension, 100)
            return quant
        else:
            return faiss.IndexFlatIP(self.embedding_dimension)

    def _normalize(self, embs: np.ndarray) -> np.ndarray:
        if self.metric == "cosine":
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return embs / norms
        return embs

    def add_embeddings(self, embeddings: np.ndarray, chunk_ids: List[str],
                       texts: List[str], metadata: List[Dict[str, Any]]):
        normalized = self._normalize(embeddings.astype(np.float32))
        current = self.index.ntotal
        if self.index_type != "flat" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(normalized)
        self.index.add(normalized)
        for i, (cid, txt, meta) in enumerate(zip(chunk_ids, texts, metadata)):
            idx = current + i
            self.chunk_ids.append(cid); self.chunk_texts.append(txt); self.chunk_metadata.append(meta)
            self.id_to_index[cid] = idx; self.index_to_id[idx] = cid
        logger.info(f"Added {len(chunk_ids)} items; total now: {self.index.ntotal}")

    def search(self, query_emb: np.ndarray, k=5, threshold: Optional[float] = None) -> List[RetrievalResult]:
        if self.index.ntotal == 0:
            return []
        qn = self._normalize(query_emb.reshape(1, -1).astype(np.float32))
        dists, inds = self.index.search(qn, min(k, self.index.ntotal))
        results = []
        for dist, idx in zip(dists[0], inds[0]):
            if idx < 0:
                continue
            score = float(dist) if self.metric == "cosine" else 1.0/(1.0+dist)
            if threshold is not None and score < threshold:
                continue
            cid = self.index_to_id.get(idx, f"unknown_{idx}")
            results.append(RetrievalResult(
                chunk_id=cid,
                content=self.chunk_texts[idx],
                score=score,
                metadata=self.chunk_metadata[idx],
                chunk_index=idx
            ))
        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Search returned {len(results)} results")
        return results

    def search_by_text(self, text: str, embedder, k=5, threshold=None):
        emb = embedder.encode_query(text)
        return self.search(emb, k=k, threshold=threshold)

    def save_index(self, path: str = None) -> str:
        idx = path or str(self.store_dir / "faiss_index.idx")
        faiss.write_index(self.index, idx)
        meta = {
            "chunk_ids": self.chunk_ids,
            "chunk_texts": self.chunk_texts,
            "chunk_metadata": self.chunk_metadata,
            "id_to_index": self.id_to_index,
            "index_to_id": {str(k): v for k, v in self.index_to_id.items()},
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "metric": self.metric
        }
        j = Path(idx).with_suffix(".json")
        p = Path(idx).with_suffix(".pkl")
        j.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        with open(p, "wb") as f:
            pickle.dump(meta, f)
        logger.info(f"Saved index to {idx}")
        return idx

    def load_index(self, path: str) -> bool:
        p = Path(path)
        if not p.exists():
            logger.error(f"Index file missing: {path}")
            return False
        self.index = faiss.read_index(str(p))
        pkl = p.with_suffix(".pkl")
        js = p.with_suffix(".json")
        meta = None
        if pkl.exists():
            with open(pkl, "rb") as f:
                meta = pickle.load(f)
        elif js.exists():
            meta = json.loads(js.read_text(encoding='utf-8'))
        if not meta:
            logger.error("Metadata missing")
            return False
        self.chunk_ids = meta['chunk_ids']; self.chunk_texts = meta['chunk_texts']
        self.chunk_metadata = meta['chunk_metadata']
        self.id_to_index = meta['id_to_index']
        self.index_to_id = {int(k): v for k, v in meta['index_to_id'].items()}
        logger.info(f"Loaded index with {self.index.ntotal} items")
        return True

    def get_statistics(self) -> Dict[str, Any]:
        mb = 0.0
        if self.index.ntotal:
            emb_bytes = self.embedding_dimension * self.index.ntotal * 4
            mb = emb_bytes / (1024 * 1024)
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "total_chunks": len(self.chunk_texts),
            "index_size_mb": round(mb, 2)
        }


def create_vector_store(embedding_dimension: int, index_type="flat", metric="cosine", store_dir=None):
    vs = FAISSVectorStore(embedding_dimension, index_type, metric, store_dir)
    idx = Path(vs.store_dir) / "faiss_index.idx"
    if idx.exists():
        vs.load_index(str(idx))
    return vs


def build_vector_store_from_embeddings(embedding_data: Dict[str, Any], store_dir=None):
    vs = create_vector_store(embedding_data['embedding_dimension'], store_dir=store_dir)
    vs.add_embeddings(embedding_data['embeddings'], embedding_data['chunk_ids'],
                      embedding_data['texts'], embedding_data['metadata'])
    return vs
