"""Dual-corpus ingestion for the RAG co-pilot.

  - **literature** — PubMed abstracts via NCBI E-utilities (esearch + efetch),
    sentence-aware chunking, cited by PMID. (Port of the WntHub 3-part pipeline,
    condensed into one async module.)
  - **tutorials**  — methods docs (scanpy/Seurat/best-practices) fetched from URLs
    or read from local files, section-aware chunking, cited by title/section/url.

Both embed via DeepInfra (``settings.RAG_EMBED_MODEL``) and upsert into pgvector
(``core/rag/store.py``). Run as a CLI:

    python -m scview.core.rag.ingest init
    python -m scview.core.rag.ingest literature --term "single-cell RNA-seq clustering" --max 100
    python -m scview.core.rag.ingest tutorials --url https://… --url https://…
    python -m scview.core.rag.ingest tutorials --file /path/to/doc.md
    python -m scview.core.rag.ingest status
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
from typing import Any

import httpx

from scview.config import get_settings
from scview.core.rag import store
from scview.core.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_EMBED_BATCH = 64

# Default methods-corpus seed queries (single-cell *methods*, not biology).
DEFAULT_LIT_TERMS = [
    "single-cell RNA-seq quality control best practices",
    "single-cell RNA-seq normalization methods",
    "single-cell RNA-seq clustering resolution benchmark",
    "single-cell RNA-seq batch effect integration Harmony",
    "single-cell RNA-seq doublet detection",
    "single-cell RNA-seq cell type annotation marker genes",
]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    # naive but robust sentence split
    return [s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if s]


def chunk_sentences(text: str, max_chars: int = 900, overlap_chars: int = 150) -> list[str]:
    """Sentence-aware chunks (~225 tokens) with small overlap — for abstracts."""
    sents = _split_sentences(text)
    chunks: list[str] = []
    cur = ""
    for s in sents:
        if cur and len(cur) + len(s) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = (cur[-overlap_chars:] + " " + s) if overlap_chars else s
        else:
            cur = (cur + " " + s) if cur else s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def chunk_sections(text: str, max_chars: int = 1100, overlap_chars: int = 150) -> list[str]:
    """Section-aware chunks for tutorial prose, with a hard size guarantee.

    The embedding model (bge, 512-token context) rejects oversized inputs, and
    scraped HTML often collapses into a few huge blocks — so we pack on any line
    break and hard-split any single piece that still exceeds ``max_chars``.
    """
    # Split on blank lines / markdown headers first, then on single newlines,
    # then hard-window anything still too large — guarantees every piece ≤ max_chars.
    raw_pieces = re.split(r"\n\s*\n|\n#{1,6}\s", text)
    pieces: list[str] = []
    for p in raw_pieces:
        for line in (p.split("\n") if len(p) > max_chars else [p]):
            line = line.strip()
            if not line:
                continue
            while len(line) > max_chars:
                pieces.append(line[:max_chars])
                line = line[max_chars - overlap_chars:]
            if line:
                pieces.append(line)

    chunks: list[str] = []
    cur = ""
    for b in pieces:
        if cur and len(cur) + len(b) + 1 > max_chars:
            chunks.append(cur.strip())
            cur = (cur[-overlap_chars:] + " " + b) if overlap_chars else b
        else:
            cur = (cur + " " + b) if cur else b
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


# ---------------------------------------------------------------------------
# Literature corpus (PubMed)
# ---------------------------------------------------------------------------


async def fetch_pubmed(term: str, retmax: int = 100) -> list[dict[str, Any]]:
    """esearch + efetch a PubMed term → list of abstract records."""
    async with httpx.AsyncClient(timeout=60) as http:
        es = await http.get(
            f"{NCBI_EUTILS}/esearch.fcgi",
            params={"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json"},
        )
        es.raise_for_status()
        ids = es.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        ef = await http.get(
            f"{NCBI_EUTILS}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml"},
        )
        ef.raise_for_status()
    return _parse_pubmed_xml(ef.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)
    out: list[dict[str, Any]] = []
    for art in root.findall(".//PubmedArticle"):
        pmid = art.findtext(".//PMID") or ""
        title = "".join(art.find(".//ArticleTitle").itertext()) if art.find(".//ArticleTitle") is not None else ""
        abstract = " ".join(
            "".join(node.itertext()) for node in art.findall(".//Abstract/AbstractText")
        ).strip()
        if not abstract:
            continue
        year = art.findtext(".//JournalIssue/PubDate/Year") or art.findtext(".//PubDate/Year") or ""
        journal = art.findtext(".//Journal/Title") or ""
        authors = []
        for a in art.findall(".//AuthorList/Author")[:3]:
            ln = a.findtext("LastName")
            if ln:
                authors.append(ln)
        author_str = (authors[0] + " et al." if len(authors) > 1 else (authors[0] if authors else ""))
        out.append({
            "pmid": pmid, "title": title.strip(), "abstract": abstract,
            "authors": author_str, "year": year, "journal": journal,
        })
    return out


async def ingest_literature(terms: list[str], retmax: int = 100) -> int:
    """Fetch, chunk, embed, and upsert PubMed abstracts for the given terms."""
    settings = get_settings()
    _require(settings)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for term in terms:
        recs = await fetch_pubmed(term, retmax)
        logger.info("literature: %d records for %r", len(recs), term)
        for r in recs:
            if r["pmid"] in seen:
                continue
            seen.add(r["pmid"])
            for i, ch in enumerate(chunk_sentences(f"{r['title']}. {r['abstract']}")):
                rows.append({
                    "doc_id": r["pmid"], "chunk_idx": i, "content": ch,
                    "metadata": {
                        "pmid": r["pmid"], "title": r["title"], "authors": r["authors"],
                        "year": r["year"], "journal": r["journal"],
                        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/",
                    },
                })
        await asyncio.sleep(0.4)  # be polite to NCBI (≤3 req/s)
    return await _embed_and_upsert(settings, "literature", rows)


# ---------------------------------------------------------------------------
# Tutorials corpus (docs)
# ---------------------------------------------------------------------------


def _slug(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


async def fetch_url_text(url: str) -> tuple[str, str]:
    """Fetch a URL and return ``(title, text)`` with HTML stripped."""
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
        r = await http.get(url, headers={"User-Agent": "scView-RAG/0.1"})
        r.raise_for_status()
    html = r.text
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title = (soup.title.string if soup.title else url) or url
        main = soup.find("main") or soup.find("article") or soup.body or soup
        text = main.get_text("\n")
    except Exception:  # bs4 missing → crude strip
        title = url
        text = re.sub(r"<[^>]+>", " ", html)
    return title.strip(), text


async def ingest_tutorials(
    urls: list[str] | None = None, files: list[str] | None = None
) -> int:
    """Ingest tutorial/methods docs from URLs and/or local files."""
    settings = get_settings()
    _require(settings)
    rows: list[dict[str, Any]] = []

    for url in urls or []:
        title, text = await fetch_url_text(url)
        slug = _slug(url)
        for i, ch in enumerate(chunk_sections(text)):
            rows.append({
                "doc_id": slug, "chunk_idx": i, "content": ch,
                "metadata": {"slug": slug, "title": title, "url": url, "section": f"part{i + 1}"},
            })
        logger.info("tutorials: %s → %d chunks", url, i + 1 if rows else 0)

    for path in files or []:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        title = path.rsplit("/", 1)[-1]
        slug = _slug(path)
        for i, ch in enumerate(chunk_sections(text)):
            rows.append({
                "doc_id": slug, "chunk_idx": i, "content": ch,
                "metadata": {"slug": slug, "title": title, "url": "", "section": f"part{i + 1}"},
            })

    return await _embed_and_upsert(settings, "tutorials", rows)


# ---------------------------------------------------------------------------
# Shared embed + upsert
# ---------------------------------------------------------------------------


async def _embed_and_upsert(settings, corpus: str, rows: list[dict[str, Any]]) -> int:
    if not rows:
        logger.warning("%s: nothing to ingest", corpus)
        return 0
    total = 0
    # Defensive cap: bge has a 512-token context. Dense technical text runs
    # ~2.9 chars/token, so ~1300 chars stays safely under 512 tokens.
    for r in rows:
        if len(r["content"]) > 1300:
            r["content"] = r["content"][:1300]
    for start in range(0, len(rows), _EMBED_BATCH):
        batch = rows[start:start + _EMBED_BATCH]
        vecs = await embed_texts(
            [r["content"] for r in batch], settings.DEEPINFRA_API_KEY, settings.RAG_EMBED_MODEL
        )
        for r, v in zip(batch, vecs):
            r["embedding"] = v
        total += await store.upsert_chunks(settings.RAG_DATABASE_URL, corpus, batch)
        logger.info("%s: upserted %d/%d", corpus, total, len(rows))
    return total


def _require(settings) -> None:
    if not settings.RAG_DATABASE_URL:
        raise SystemExit("RAG_DATABASE_URL is not set — add a Neon/Postgres DSN to .env")
    if not settings.DEEPINFRA_API_KEY:
        raise SystemExit("DEEPINFRA_API_KEY is not set — embeddings require it")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="scView RAG corpus ingestion")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create pgvector schema")
    sub.add_parser("status", help="show per-corpus chunk counts")

    lit = sub.add_parser("literature", help="ingest PubMed abstracts")
    lit.add_argument("--term", action="append", help="PubMed query (repeatable)")
    lit.add_argument("--max", type=int, default=100, help="max abstracts per term")

    tut = sub.add_parser("tutorials", help="ingest tutorial/methods docs")
    tut.add_argument("--url", action="append", default=[], help="doc URL (repeatable)")
    tut.add_argument("--file", action="append", default=[], help="local doc file (repeatable)")

    args = p.parse_args()
    settings = get_settings()

    async def _run() -> None:
        if args.cmd == "init":
            await store.init_db(settings.RAG_DATABASE_URL, settings.RAG_EMBED_DIM)
            print("schema ready")
        elif args.cmd == "status":
            print(await store.corpus_counts(settings.RAG_DATABASE_URL))
        elif args.cmd == "literature":
            n = await ingest_literature(args.term or DEFAULT_LIT_TERMS, args.max)
            print(f"literature: {n} chunks ingested")
        elif args.cmd == "tutorials":
            n = await ingest_tutorials(args.url, args.file)
            print(f"tutorials: {n} chunks ingested")
        await store.close_pool()

    asyncio.run(_run())


if __name__ == "__main__":
    main()
