from __future__ import annotations

import re
from typing import List, Dict, Any

# Regex utilities
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')
_WORD = re.compile(r"\b\w+\b", re.UNICODE)


def normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace and trim ends."""
    return re.sub(r"\s+", " ", s or "").strip()


def split_sentences(text: str) -> List[str]:
    """
    Deterministic sentence splitter:
    - Splits on [.?!] followed by whitespace
    - No cross-sentence merging (leave that to chunking)
    """
    text = normalize_whitespace(text)
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def chunk_by_chars(
    text: str,
    max_chars: int = 1200,
    overlap: int = 120
) -> List[Dict[str, Any]]:
    """
    Build chunks ~max_chars, aligned to sentence boundaries, with soft overlap.
    Returns list of {text, start_char, end_char}.
    """
    if not text:
        return []

    sents = split_sentences(text)
    chunks: List[Dict[str, Any]] = []

    cur: List[str] = []
    cur_len = 0
    start = 0
    pos = 0

    for s in sents:
        s_len = len(s) + 1  # account for an intervening space
        if cur and (cur_len + s_len > max_chars):
            chunk_text = " ".join(cur).strip()
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": start + len(chunk_text),
                }
            )
            # Overlap tail to preserve context continuity
            tail = ""
            if overlap > 0 and len(chunk_text) > overlap:
                tail = chunk_text[-overlap:]
            cur = [tail] if tail else []
            cur_len = len(tail)
            start = pos - len(tail)

        cur.append(s)
        cur_len += s_len
        pos += s_len

    if cur:
        chunk_text = " ".join(cur).strip()
        chunks.append(
            {
                "text": chunk_text,
                "start_char": start,
                "end_char": start + len(chunk_text),
            }
        )

    return chunks


def tokenize_lower(s: str) -> List[str]:
    """Lowercased token stream for lexical scoring."""
    return [m.group(0).lower() for m in _WORD.finditer(s or "")]
