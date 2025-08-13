from __future__ import annotations
from typing import List, Dict, Tuple
import math
import numpy as np
from app.textproc import tokenize_lower

def bm25_scores(query: str, docs: List[str], k1: float = 1.2, b: float = 0.75) -> List[float]:
    """
    Minimal BM25 over whitespace/word tokens.
    """
    # tokenize
    q_tokens = tokenize_lower(query)
    D_tokens = [tokenize_lower(d) for d in docs]
    N = len(D_tokens)
    if N == 0:
        return []
    dl = [len(dt) for dt in D_tokens]
    avgdl = sum(dl) / max(N, 1)

    # document frequencies
    df: Dict[str, int] = {}
    for dt in D_tokens:
        for t in set(dt):
            df[t] = df.get(t, 0) + 1

    def idf(t: str) -> float:
        # bm25+style idf with +1 to avoid zero/negatives
        n = df.get(t, 0)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    # term frequencies per doc
    scores = [0.0] * N
    for i, dt in enumerate(D_tokens):
        tf: Dict[str, int] = {}
        for t in dt:
            tf[t] = tf.get(t, 0) + 1
        denom = k1 * (1 - b + b * (dl[i] / max(avgdl, 1e-9)))
        s = 0.0
        for t in q_tokens:
            if t not in tf:
                continue
            s += idf(t) * (tf[t] * (k1 + 1)) / (tf[t] + denom)
        scores[i] = s
    return scores

def minmax(x: List[float]) -> List[float]:
    if not x:
        return x
    lo, hi = min(x), max(x)
    if hi - lo < 1e-12:
        return [0.0 for _ in x]
    return [(v - lo) / (hi - lo) for v in x]

def hybrid_scores(query: str, docs: List[str], cos_scores: List[float] | None, alpha: float = 0.5) -> List[float]:
    """
    Combine BM25 (lexical) and cosine (semantic) via min-max normalized convex combo.
    alpha: weight on cosine (0..1). 0.5 is a good default.
    """
    bm = bm25_scores(query, docs)
    bm_n = minmax(bm)
    if cos_scores is None:
        return bm_n
    cos_n = minmax(cos_scores)
    return [alpha * cs + (1 - alpha) * bs for cs, bs in zip(cos_n, bm_n)]

import datetime

def freshness_score(year: int | None, now_year: int | None = None, half_life_years: float = 5.0) -> float:
    """
    Exponential recency: score = 0..1, where 'now' is ~1 and older decays with half-life.
    """
    if year is None:
        return 0.5  # neutral if unknown
    if now_year is None:
        now_year = datetime.datetime.utcnow().year
    age = max(0.0, float(now_year - year))
    # exp2(-age/half_life): 1 at age=0, 0.5 at half_life, etc.
    return 2.0 ** (-age / max(half_life_years, 0.1))

def blend_with_freshness(content_scores: List[float], fresh_scores: List[float], freshness_weight: float = 0.3) -> List[float]:
    """
    Combine content relevance (BM25/Embeddings hybrid) with freshness via convex combo.
    """
    from .ranking import minmax  # re-use normalization
    c = minmax(content_scores)
    f = minmax(fresh_scores)
    fw = max(0.0, min(1.0, freshness_weight))
    return [ (1.0 - fw) * ci + fw * fi for ci, fi in zip(c, f) ]
