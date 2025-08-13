# app/routers_answer.py
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.config import settings
from app.ncbi import NCBIClient
from app.textproc import chunk_by_chars
from app.embedding import embed_texts, cosine_matrix
from app.ranking import hybrid_scores, freshness_score, blend_with_freshness
from app.evidence import classify_study_type, preference_boost, section_boost
from app.synthesis import build_messages, call_openai_json, unique_references

router = APIRouter(tags=["answer"])


class AnswerRequest(BaseModel):
    # Core
    question: str = Field(..., description="User question to answer using PubMed/PMC evidence only")
    term: str = Field(..., description="PubMed query (e.g., disease[tiab])")
    retmax: int = Field(30, ge=1, le=100, description="Max PubMed records to retrieve")

    # Chunking
    chunk_chars: int = Field(1200, ge=300, le=4000, description="Approx max chars per chunk")
    overlap: int = Field(120, ge=0, le=800, description="Soft overlap (chars) between chunks")

    # Ranking (content)
    top_k: int = Field(8, ge=1, le=20, description="Top evidence chunks to use")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight of embeddings in hybrid scoring (0=BM25 only)")
    use_embeddings: bool = Field(True, description="Use OpenAI embeddings to add semantic scoring")

    # Ranking (freshness + preferences)
    freshness_weight: float = Field(0.3, ge=0.0, le=1.0, description="Blend weight for recency")
    half_life_years: float = Field(5.0, ge=0.1, le=50.0, description="Half-life for recency decay in years")
    prefer_types: Optional[str] = Field(
        None,
        description="Comma-separated study types to up-weight (e.g., 'RCT,Meta-analysis')",
    )

    # Full-text (PMC)
    want_fulltext: bool = Field(True, description="Also retrieve open-access PMC full text if available")
    include_sections: str = Field(
        "Results,Methods,Discussion",
        description="Comma-separated PMC sections to include (e.g., 'Results,Methods,Discussion,Conclusion')",
    )


# Convenience helper so accidental GETs don’t 405
@router.get("/answer")
async def grounded_answer_get():
    return {
        "detail": "This endpoint requires POST with a JSON body.",
        "how_to": "Use Swagger UI at /docs → POST /pubmed/answer → Try it out.",
        "example_body": {
            "question": "Do statins reduce all-cause mortality in high-risk adults?",
            "term": "statins mortality randomized[tiab]",
            "retmax": 30,
            "chunk_chars": 1200,
            "overlap": 120,
            "top_k": 6,
            "alpha": 0.5,
            "freshness_weight": 0.35,
            "half_life_years": 6,
            "prefer_types": "RCT,Meta-analysis",
            "want_fulltext": True,
            "include_sections": "Results,Methods,Discussion",
            "use_embeddings": True,
        },
    }


# Grounded synthesis: retrieve → chunk → rank (hybrid + freshness + preferences + PMC sections) → synthesize
@router.post("/answer")
async def grounded_answer(req: AnswerRequest):
    # 1) Retrieval primitives
    client = NCBIClient(
        api_key=settings.ncbi_api_key,
        email=settings.ncbi_email,
        tool=settings.ncbi_tool,
    )
    pmids = await client.esearch(term=req.term, retmax=req.retmax, sort="relevance")
    if not pmids:
        return {
            "used_embeddings": False,
            "answer": "insufficient_evidence",
            "citations": [],
            "notes": "No PubMed hits.",
            "references": [],
        }

    meta = await client.esummary(pmids)
    xml = await client.efetch_pubmed_xml(pmids)
    abstracts = client.parse_pubmed_abstracts(xml)
    records = client.assemble_records(pmids, meta, abstracts)

    # 2) Build chunk corpus (Abstracts + optional PMC sections)
    now_year = datetime.datetime.utcnow().year
    prefs = [t.strip() for t in (req.prefer_types.split(",") if req.prefer_types else []) if t.strip()]
    wanted_secs = [s.strip().capitalize() for s in req.include_sections.split(",") if s.strip()]

    corpus: List[str] = []
    chunk_meta: List[Dict[str, Any]] = []

    # 2a) Abstract chunks
    for r in records:
        if not r["abstract"]:
            continue
        stype = classify_study_type(r.get("pubtypes") or [], r.get("title", ""))
        parts = chunk_by_chars(r["abstract"], max_chars=req.chunk_chars, overlap=req.overlap)
        for idx, p in enumerate(parts):
            corpus.append(p["text"])
            chunk_meta.append(
                {
                    "source": "pubmed",
                    "pmid": r["pmid"],
                    "pmcid": None,
                    "section": "Abstract",
                    "title": r["title"],
                    "journal": r["journal"],
                    "pubdate": r["pubdate"],
                    "year": r.get("year"),
                    "pubtypes": r.get("pubtypes") or [],
                    "study_type": stype,
                    "doi": r["doi"],
                    "chunk_id": f"{r['pmid']}-abs-{idx}",
                    "text": p["text"],  # kept for prompt assembly
                }
            )

    # 2b) PMC full text sections (optional)
    if req.want_fulltext:
        pmc_map = await client.elink_pmc(pmids)  # PMID -> PMCID (numeric string, no 'PMC' prefix)
        pmcids = list({v for v in pmc_map.values()})
        if pmcids:
            pmc_xml = await client.efetch_pmc_xml(pmcids)
            pmc_sec_map = client.parse_pmc_sections(pmc_xml)  # { pmcid -> {SectionName -> text} }
            for pmid, pmcid in pmc_map.items():
                sec_dict = pmc_sec_map.get(pmcid, {})
                if not sec_dict:
                    continue
                rmeta = next((r for r in records if r["pmid"] == pmid), None)
                stype = classify_study_type(rmeta.get("pubtypes") or [], rmeta.get("title", "")) if rmeta else "Unspecified"
                for sec_name, sec_text in sec_dict.items():
                    if sec_name not in wanted_secs:
                        continue
                    parts = chunk_by_chars(sec_text, max_chars=req.chunk_chars, overlap=req.overlap)
                    for idx, p in enumerate(parts):
                        corpus.append(p["text"])
                        chunk_meta.append(
                            {
                                "source": "pmc",
                                "pmid": pmid,
                                "pmcid": pmcid,  # numeric (no 'PMC' prefix)
                                "section": sec_name,
                                "title": (rmeta or {}).get("title", ""),
                                "journal": (rmeta or {}).get("journal", ""),
                                "pubdate": (rmeta or {}).get("pubdate", ""),
                                "year": (rmeta or {}).get("year"),
                                "pubtypes": (rmeta or {}).get("pubtypes") or [],
                                "study_type": stype,
                                "doi": (rmeta or {}).get("doi", ""),
                                "chunk_id": f"{pmcid}-{sec_name}-{idx}",
                                "text": p["text"],
                            }
                        )

    if not corpus:
        return {
            "used_embeddings": False,
            "answer": "insufficient_evidence",
            "citations": [],
            "notes": "No abstracts or full-text sections found for grounding.",
            "references": [],
        }

    # 3) Ranking — hybrid lexical+semantic → freshness blend → study-type + section boosts
    cos_scores = None
    used_embeddings = False
    if req.use_embeddings:
        embs = await embed_texts([req.term] + corpus)
        if embs is not None:
            cos_scores = cosine_matrix(embs[0:1], embs[1:])[0].tolist()
            used_embeddings = True

    scores = hybrid_scores(req.term, corpus, cos_scores, alpha=req.alpha)

    fresh = [freshness_score(meta_i.get("year"), now_year, req.half_life_years) for meta_i in chunk_meta]
    scores = blend_with_freshness(scores, fresh, req.freshness_weight)

    scores = [
        sc * preference_boost(chunk_meta[i]["study_type"], prefs) * section_boost(chunk_meta[i]["section"])
        for i, sc in enumerate(scores)
    ]

    order = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    top_idxs = order[: max(1, min(req.top_k, len(order)))]

    top_chunks: List[Dict[str, Any]] = []
    for i in top_idxs:
        meta_i = dict(chunk_meta[i])
        meta_i["text"] = corpus[i]  # include the actual snippet for prompting
        top_chunks.append(meta_i)

    # 4) Synthesis (retrieve-or-refuse; JSON-mode contract)
    messages = build_messages(req.question, top_chunks)
    model_obj = await call_openai_json(messages)

    # 5) References bundle for UI (PubMed + PMC links)
    references = unique_references(top_chunks)

    # 6) Contracted response
    return {
        "used_embeddings": used_embeddings,
        "answer": model_obj.get("answer", "insufficient_evidence"),
        "citations": model_obj.get("citations", []),
        "notes": model_obj.get("notes", ""),
        "references": references,
    }
