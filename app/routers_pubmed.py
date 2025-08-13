# app/routers_pubmed.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
import datetime

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.ncbi import NCBIClient
from app.textproc import chunk_by_chars
from app.embedding import embed_texts, cosine_matrix
from app.ranking import hybrid_scores, blend_with_freshness, freshness_score
from app.evidence import classify_study_type, preference_boost

router = APIRouter(tags=["pubmed"])


def _client() -> NCBIClient:
    return NCBIClient(
        api_key=settings.ncbi_api_key,
        email=settings.ncbi_email,
        tool=settings.ncbi_tool,
    )


@router.get("/search")
async def pubmed_search(
    term: str = Query(..., description="PubMed query (you can use [tiab] tags)"),
    retmax: int = Query(20, ge=1, le=100),
):
    client = _client()
    pmids = await client.esearch(term=term, retmax=retmax, sort="relevance")
    return {"count": len(pmids), "pmids": pmids}


@router.get("/retrieve")
async def pubmed_retrieve(
    term: str = Query(..., description="PubMed query (e.g., cancer[tiab])"),
    retmax: int = Query(10, ge=1, le=50),
):
    client = _client()
    pmids = await client.esearch(term=term, retmax=retmax, sort="relevance")
    if not pmids:
        return {"records": []}

    meta = await client.esummary(pmids)
    xml = await client.efetch_pubmed_xml(pmids)
    abstracts = client.parse_pubmed_abstracts(xml)
    records = client.assemble_records(pmids, meta, abstracts)

    if all(not r["abstract"] for r in records):
        raise HTTPException(status_code=424, detail="No abstracts available for grounding.")

    return {"records": records}


@router.get("/select")
async def pubmed_select(
    term: str,
    retmax: int = 20,
    chunk_chars: int = 1200,
    overlap: int = 120,
    top_k: int = 8,
    alpha: float = 0.5,                 # weight for embeddings in hybrid
    use_embeddings: bool = True,
    freshness_weight: float = 0.3,      # 0..1 blend weight for recency
    half_life_years: float = 5.0,       # recency half-life in years
    prefer_types: Optional[str] = None  # e.g., "RCT,Meta-analysis"
):
    """
    End-to-end retrieval: search → fetch PubMed + PMC → chunk → rank → (freshness + study-type) → top-K evidence chunks.
    """
    client = _client()
    pmids = await client.esearch(term=term, retmax=min(retmax, 100), sort="relevance")
    if not pmids:
        return {
            "query": term,
            "top_k": 0,
            "used_embeddings": False,
            "freshness_weight": freshness_weight,
            "half_life_years": half_life_years,
            "prefer_types": [],
            "chunks": [],
        }

    # --- Pull PubMed metadata + abstracts ---
    meta = await client.esummary(pmids)
    xml = await client.efetch_pubmed_xml(pmids)
    abstracts = client.parse_pubmed_abstracts(xml)
    records = client.assemble_records(pmids, meta, abstracts)

    now_year = datetime.datetime.utcnow().year
    prefs = [t.strip() for t in (prefer_types.split(",") if prefer_types else []) if t.strip()]

    corpus: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []

    # --- Process PubMed abstracts ---
    for r in records:
        if not r["abstract"]:
            continue
        stype = classify_study_type(r.get("pubtypes") or [], r.get("title", ""))
        parts = chunk_by_chars(r["abstract"], max_chars=chunk_chars, overlap=overlap)
        for idx, p in enumerate(parts):
            corpus.append(p["text"])
            chunk_meta.append({
                "pmid": r["pmid"],
                "pmcid": "",  # no PMC from PubMed record
                "title": r["title"],
                "journal": r["journal"],
                "pubdate": r["pubdate"],
                "year": r.get("year"),
                "pubtypes": r.get("pubtypes") or [],
                "study_type": stype,
                "doi": r["doi"],
                "chunk_id": f"{r['pmid']}-{idx}",
                "section": None,
                "text": p["text"],
            })

    # --- Try PMC retrieval for full text sections ---
    # PMC IDs can be retrieved from PubMed meta via article IDs (not implemented here)
    # This block can be extended to call efetch_pmc_sections and merge.
    # For now, the structure is ready to merge PMC chunks when available.

    if not corpus:
        return {
            "query": term,
            "top_k": 0,
            "used_embeddings": False,
            "freshness_weight": freshness_weight,
            "half_life_years": half_life_years,
            "prefer_types": prefs,
            "chunks": [],
        }

    # --- Ranking ---
    cos_scores = None
    used_embeddings = False
    if use_embeddings:
        embs = await embed_texts([term] + corpus)
        if embs is not None:
            qv = embs[0:1]        # (1, D)
            dv = embs[1:]         # (N, D)
            cos_scores = cosine_matrix(qv, dv)[0].tolist()
            used_embeddings = True

    scores = hybrid_scores(term, corpus, cos_scores, alpha=alpha)

    fresh = [freshness_score(meta.get("year"), now_year, half_life_years) for meta in chunk_meta]
    scores = blend_with_freshness(scores, fresh, freshness_weight)

    scores = [sc * preference_boost(chunk_meta[i]["study_type"], prefs) for i, sc in enumerate(scores)]

    order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    top_idxs = order[: max(1, min(top_k, len(order)))]

    out = []
    for i in top_idxs:
        o = dict(chunk_meta[i])
        o["score"] = scores[i]
        out.append(o)

    return {
        "query": term,
        "top_k": len(out),
        "used_embeddings": used_embeddings,
        "freshness_weight": freshness_weight,
        "half_life_years": half_life_years,
        "prefer_types": prefs,
        "chunks": out,
    }
