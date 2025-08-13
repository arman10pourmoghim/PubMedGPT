from __future__ import annotations
from typing import List, Optional
import httpx
import numpy as np
from app.config import settings

# You can swap this with a larger model later for quality
_EMBEDDING_MODEL = "text-embedding-3-small"

async def embed_texts(texts: List[str]) -> Optional[np.ndarray]:
    """
    Returns an array of shape (N, D) or None if embedding is unavailable.
    """
    if not settings.openai_api_key:
        return None
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {settings.openai_api_key}"}
    payload = {"model": _EMBEDDING_MODEL, "input": texts}
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()["data"]
            mat = np.array([row["embedding"] for row in data], dtype=float)
            return mat
    except Exception:
        # Fail closed to BM25-only mode
        return None

def cosine_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    return np.einsum("id,jd->ij", qn, Mn)  # (n_queries x n_docs)
