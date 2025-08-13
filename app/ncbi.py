# app/ncbi.py
from __future__ import annotations

import time
import re
from typing import Any, Dict, List, Optional

import httpx
from lxml import etree
from tenacity import retry, wait_exponential_jitter, stop_after_attempt

from app.metrics import metrics
from app.cache import default_cache, short_cache

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


class NCBIClient:
    """
    Thin, resilient wrapper around NCBI E-utilities (PubMed / PMC).
    - Attaches etiquette params (tool, email, api_key) to every call
    - Retries with jittered exponential backoff on transient failures
    - Caches common calls (ESearch/ESummary/EFetch/Elink) with TTL
    - Emits lightweight metrics for ops visibility
    """

    def __init__(self, api_key: Optional[str], email: Optional[str], tool: Optional[str]):
        self.api_key = api_key or ""
        self.email = email or ""
        self.tool = tool or "pubmed-gpt-app"
        self.headers = {"User-Agent": f"{self.tool} ({self.email})"}

    # ----------------------------- internals -----------------------------

    def _params_core(self) -> Dict[str, str]:
        base = {"tool": self.tool, "email": self.email}
        if self.api_key:
            base["api_key"] = self.api_key
        return base

    @retry(wait=wait_exponential_jitter(initial=0.5, max=8), stop=stop_after_attempt(5))
    async def _get_json(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        r = await client.get(url, params={**self._params_core(), **params}, headers=self.headers, timeout=30)
        r.raise_for_status()
        return r.json()

    @retry(wait=wait_exponential_jitter(initial=0.5, max=8), stop=stop_after_attempt(5))
    async def _get_text(self, client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> str:
        r = await client.get(url, params={**self._params_core(), **params}, headers=self.headers, timeout=60)
        r.raise_for_status()
        return r.text

    # ----------------------------- PubMed -----------------------------

    async def esearch(self, term: str, retmax: int = 20, sort: str = "relevance") -> List[str]:
        """
        Search PubMed and return a list of PMIDs.
        Cached short TTL (volatile).
        """
        url = EUTILS_BASE + "esearch.fcgi"
        params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": str(retmax), "sort": sort}
        cache_key = ("esearch", term, retmax, sort, bool(self.api_key))

        t0 = time.perf_counter()
        cached = await short_cache.get(cache_key)
        if cached is not None:
            metrics.inc("cache.hit.esearch")
            return cached

        metrics.inc("cache.miss.esearch")
        async with httpx.AsyncClient() as c:
            data = await self._get_json(c, url, params)
        pmids = data.get("esearchresult", {}).get("idlist", []) or []

        await short_cache.set(cache_key, pmids)
        metrics.observe_ms("ncbi.esearch.ms", (time.perf_counter() - t0) * 1000)
        metrics.inc("ncbi.esearch.count")
        return pmids

    async def esummary(self, pmids: List[str]) -> Dict[str, Any]:
        """
        Fetch PubMed metadata (titles, journal, pubdate, doi, pubtypes, etc.).
        Cached medium TTL.
        """
        if not pmids:
            return {}
        url = EUTILS_BASE + "esummary.fcgi"
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "json"}
        cache_key = ("esummary", tuple(pmids), bool(self.api_key))

        t0 = time.perf_counter()
        cached = await default_cache.get(cache_key)
        if cached is not None:
            metrics.inc("cache.hit.esummary")
            return cached

        metrics.inc("cache.miss.esummary")
        async with httpx.AsyncClient() as c:
            data = await self._get_json(c, url, params)
        res = data.get("result", {})

        await default_cache.set(cache_key, res)
        metrics.observe_ms("ncbi.esummary.ms", (time.perf_counter() - t0) * 1000)
        metrics.inc("ncbi.esummary.count")
        return res

    async def efetch_pubmed_xml(self, pmids: List[str]) -> str:
        """
        Fetch PubMed abstracts as XML (EFetch).
        Cached medium TTL.
        """
        if not pmids:
            return ""
        url = EUTILS_BASE + "efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}
        cache_key = ("efetch.pubmed", tuple(pmids))

        t0 = time.perf_counter()
        cached = await default_cache.get(cache_key)
        if cached is not None:
            metrics.inc("cache.hit.efetch_pubmed")
            return cached

        metrics.inc("cache.miss.efetch_pubmed")
        async with httpx.AsyncClient() as c:
            xml_text = await self._get_text(c, url, params)

        await default_cache.set(cache_key, xml_text)
        metrics.observe_ms("ncbi.efetch_pubmed.ms", (time.perf_counter() - t0) * 1000)
        metrics.inc("ncbi.efetch_pubmed.count")
        return xml_text

    @staticmethod
    def parse_pubmed_abstracts(xml_text: str) -> Dict[str, str]:
        """
        Parse EFetch XML and return { pmid -> abstract_text }.
        Concatenates multiple <AbstractText> segments.
        """
        if not xml_text.strip():
            return {}
        parser = etree.XMLParser(recover=True)
        root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
        out: Dict[str, str] = {}
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            if not pmid:
                continue
            nodes = art.findall(".//Abstract/AbstractText")
            text = "\n".join(etree.tostring(n, method="text", encoding="unicode").strip() for n in nodes).strip()
            out[pmid] = text
        return out

    @staticmethod
    def assemble_records(pmids: List[str], meta: Dict[str, Any], abstracts: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Merge ESummary metadata + abstracts into normalized records.
        Adds: pubtypes (list[str]), year (int or None).
        """
        def _parse_year(s: str | None) -> int | None:
            if not s:
                return None
            m = re.search(r"\b(19|20)\d{2}\b", s)
            return int(m.group(0)) if m else None

        recs: List[Dict[str, Any]] = []
        for pmid in pmids:
            m = meta.get(pmid) or {}
            pubdate = m.get("pubdate") or m.get("epubdate") or ""
            recs.append({
                "pmid": pmid,
                "title": m.get("title") or "",
                "journal": m.get("fulljournalname") or m.get("source") or "",
                "pubdate": pubdate,
                "year": _parse_year(pubdate),
                "doi": (m.get("elocationid") or "").replace("doi: ", "") if m.get("elocationid") else "",
                "pubtypes": m.get("pubtype") or [],
                "abstract": abstracts.get(pmid, "")
            })
        return recs

    # ----------------------------- PMC (full text) -----------------------------

    async def elink_pmc(self, pmids: List[str]) -> Dict[str, str]:
        """
        Map PMID -> PMCID (numeric string without 'PMC' prefix) when open-access full text exists.
        Cached medium TTL.
        """
        if not pmids:
            return {}
        url = EUTILS_BASE + "elink.fcgi"
        params = {"dbfrom": "pubmed", "linkname": "pubmed_pmc", "id": ",".join(pmids)}
        cache_key = ("elink.pmc", tuple(pmids))

        t0 = time.perf_counter()
        cached = await default_cache.get(cache_key)
        if cached is not None:
            metrics.inc("cache.hit.elink_pmc")
            return cached

        metrics.inc("cache.miss.elink_pmc")
        async with httpx.AsyncClient() as c:
            xml = await self._get_text(c, url, params)

        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(xml.encode("utf-8"), parser=parser)
            out: Dict[str, str] = {}
            for ls in root.findall(".//LinkSet"):
                pmid = ls.findtext(".//IdList/Id")
                pmc_id = ls.findtext(".//LinkSetDb[LinkName='pubmed_pmc']/Link/Id")
                if pmid and pmc_id:
                    out[pmid] = pmc_id  # numeric part (e.g., '1234567')
        except Exception:
            out = {}

        await default_cache.set(cache_key, out)
        metrics.observe_ms("ncbi.elink_pmc.ms", (time.perf_counter() - t0) * 1000)
        metrics.inc("ncbi.elink_pmc.count")
        return out

    async def efetch_pmc_xml(self, pmcids: List[str]) -> str:
        """
        Fetch PMC full-text NXML for a list of PMCID numbers (no 'PMC' prefix in the param).
        Cached medium TTL.
        """
        if not pmcids:
            return ""
        url = EUTILS_BASE + "efetch.fcgi"
        params = {"db": "pmc", "id": ",".join(pmcids), "retmode": "xml"}
        cache_key = ("efetch.pmc", tuple(pmcids))

        t0 = time.perf_counter()
        cached = await default_cache.get(cache_key)
        if cached is not None:
            metrics.inc("cache.hit.efetch_pmc")
            return cached

        metrics.inc("cache.miss.efetch_pmc")
        async with httpx.AsyncClient() as c:
            xml_text = await self._get_text(c, url, params)

        await default_cache.set(cache_key, xml_text)
        metrics.observe_ms("ncbi.efetch_pmc.ms", (time.perf_counter() - t0) * 1000)
        metrics.inc("ncbi.efetch_pmc.count")
        return xml_text

    @staticmethod
    def parse_pmc_sections(xml_text: str) -> Dict[str, Dict[str, str]]:
        """
        Robustly extract high-signal sections per PMCID from PMC NXML.

        Returns:
          {
            'PMCID_NUM': {
              'Results': '...',
              'Methods': '...',
              'Discussion': '...',
              'Conclusion': '...',
              'Limitations': '...'
            },
            ...
          }

        Notes:
        - Namespace-agnostic (uses local-name()).
        - Normalizes PMCID keys to the numeric part (e.g., 'PMC12345' -> '12345').
        - Uses <sec sec-type> and/or <title> to identify section names.
        """
        if not xml_text or not xml_text.strip():
            return {}

        try:
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(xml_text.encode("utf-8"), parser=parser)
        except Exception:
            return {}

        wanted = ("results", "methods", "discussion", "conclusion", "limitations")
        out: Dict[str, Dict[str, str]] = {}

        # Find all <article> elements regardless of namespace
        articles = root.xpath('//*[local-name()="article"]')
        for art in articles:
            # Prefer article-id[@pub-id-type="pmcid"]
            pmcid_txts = art.xpath(
                './/*[local-name()="article-id" and translate(@pub-id-type,"PMCID","pmcid")="pmcid"]/text()'
            )
            pmcid: Optional[str] = None
            if pmcid_txts:
                pmcid = pmcid_txts[0].strip()
                # normalize to numeric only
                if pmcid.upper().startswith("PMC"):
                    pmcid = pmcid[3:]
            else:
                # fallback: any article-id text like 'PMC9999999' or '9999999'
                any_ids = [t.strip() for t in art.xpath('.//*[local-name()="article-id"]/text()') if t and t.strip()]
                for t in any_ids:
                    m = re.search(r"PMC?(\d+)", t, re.IGNORECASE)
                    if m:
                        pmcid = m.group(1)
                        break

            if not pmcid:
                # last resort: look for any <id> text with PMC pattern
                any_txts = [t.strip() for t in art.xpath('.//*[local-name()="id"]/text()') if t and t.strip()]
                for t in any_txts:
                    m = re.search(r"PMC?(\d+)", t, re.IGNORECASE)
                    if m:
                        pmcid = m.group(1)
                        break

            if not pmcid:
                # can't identify PMCID → skip article
                continue

            buckets: Dict[str, List[str]] = {k.capitalize(): [] for k in wanted}

            # Walk all <sec> regardless of nesting
            secs = art.xpath('.//*[local-name()="sec"]')
            for sec in secs:
                # Prefer @sec-type if present, else the <title> text
                sec_type = (sec.get("sec-type") or "").strip().lower()
                title_txt = " ".join(
                    t.strip() for t in sec.xpath('.//*[local-name()="title"]/text()') if t and t.strip()
                ).lower()

                # Capture all paragraph text within the section (including nested)
                para_text = " ".join(
                    t.strip() for t in sec.xpath('.//*[local-name()="p"]//text()') if t and t.strip()
                ).strip()
                if not para_text:
                    continue

                # Decide the bucket name
                chosen: Optional[str] = None
                if sec_type in wanted:
                    chosen = sec_type
                else:
                    for w in wanted:
                        if w in title_txt:
                            chosen = w
                            break

                if chosen:
                    buckets[chosen.capitalize()].append(para_text)

            # Materialize only non-empty sections
            realized = {k: "\n".join(v).strip() for k, v in buckets.items() if v}
            if realized:
                out[pmcid] = realized

        return out
