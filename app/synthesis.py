# app/synthesis.py
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

# ---- System guidance: strict grounding + JSON-mode contract ----
SYSTEM_PROMPT = (
    "You are a meticulous biomedical literature analyst. "
    "You may ONLY answer using the provided PubMed/PMC excerpts. "
    "Every non-trivial claim must be supported by inline citations like [PMID:########] or [PMCID:PMC########]. "
    "If there is insufficient evidence, respond with this exact JSON object: "
    "{\"answer\":\"insufficient_evidence\",\"citations\":[],\"notes\":\"<why>\"} "
    "and do not speculate."
)


def build_messages(question: str, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build a grounding-first prompt with compact evidence snippets.
    Each context item may include: pmid (str), pmcid (str, no 'PMC' prefix), title (str),
    text (str snippet), and optional section (e.g., 'Results').
    """
    ctx_lines: List[str] = []
    for c in contexts:
        if c.get("pmcid"):
            tag = f"[PMCID:PMC{c['pmcid']}]"
        else:
            tag = f"[PMID:{c['pmid']}]"
        title = (c.get("title") or "").strip()
        snippet = (c.get("text") or "").strip()
        section = c.get("section")
        header = f"{tag} {title}" + (f" — Section: {section}" if section else "")
        ctx_lines.append(f"{header}\n{snippet}")

    user_block = (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Evidence (snippets; cite with the PMIDs/PMCIDs shown):\n"
        + "\n\n".join(ctx_lines)
    )

    # Strict JSON response rules to keep outputs parseable and auditable.
    assistant_rules = (
        "Return a STRICT JSON object with keys: "
        "{\"answer\": string, "
        "\"citations\": [{\"pmid\": string, \"pmcid\": string, \"quote\": string}], "
        "\"notes\": string}. "
        "For each citation, include EITHER pmid OR pmcid (one is sufficient); leave the other as an empty string. "
        "Each citation must include a concise quote (<=200 chars) copied verbatim from the provided snippet. "
        "Do not invent identifiers or quotes beyond the snippets."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
        {"role": "assistant", "content": assistant_rules},
    ]


async def call_openai_json(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Call OpenAI Chat Completions with JSON mode to enforce a machine-checkable schema.
    Fails closed with an 'insufficient_evidence' object on any error or malformed output.
    """
    if not settings.openai_api_key or not settings.openai_model:
        return {
            "answer": "insufficient_evidence",
            "citations": [],
            "notes": "OpenAI credentials not configured.",
        }

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.openai_model,
        "messages": messages,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},  # JSON mode
    }

    try:
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()

        raw = data["choices"][0]["message"]["content"]
        obj = json.loads(raw)

        # Basic schema guard
        if not isinstance(obj, dict):
            raise ValueError("Model output not a JSON object")
        if "answer" not in obj or "citations" not in obj:
            raise ValueError("Missing required keys")

        # Normalize fields
        obj.setdefault("notes", "")
        citations = obj.get("citations", [])
        if not isinstance(citations, list):
            citations = []

        cleaned: List[Dict[str, str]] = []
        for citem in citations:
            pmid = str(citem.get("pmid", "") or "").strip()
            pmcid = str(citem.get("pmcid", "") or "").strip()
            quote = str(citem.get("quote", "") or "").strip()
            # Enforce “either pmid or pmcid” and require a quote
            if quote and (pmid or pmcid):
                # Normalize PMCID to numeric-without-PMC for consistency; store with PMC prefix in UI if needed
                if pmcid.upper().startswith("PMC"):
                    pmcid = pmcid[3:]
                cleaned.append({"pmid": pmid, "pmcid": pmcid, "quote": quote})

        obj["citations"] = cleaned
        return obj

    except Exception as e:
        return {
            "answer": "insufficient_evidence",
            "citations": [],
            "notes": f"OpenAI call error or malformed output: {e}",
        }


def unique_references(chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Build a de-duplicated references list from the ranked chunks.
    Prioritize PubMed URLs; include PMC URL when available.
    Output shape per item: { pmid, pmcid, title, url, pmc_url }
    """
    seen = set()
    refs: List[Dict[str, str]] = []
    for ch in chunks:
        pmid = str(ch.get("pmid", "") or "")
        pmcid = str(ch.get("pmcid", "") or "")
        key = (pmid, pmcid)
        if key in seen:
            continue
        seen.add(key)
        title = ch.get("title", "") or ""
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/" if pmcid else ""
        refs.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": title,
                "url": url,
                "pmc_url": pmc_url,
            }
        )
    return refs
