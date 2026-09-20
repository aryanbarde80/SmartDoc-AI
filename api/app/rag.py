"""
api/app/rag.py — Phase 3: hybrid retrieval.

Pipeline per question:
  1. Dense candidates:  cosine similarity over pgvector embeddings   (semantic)
  2. Lexical candidates: Postgres ts_rank over tsvector              (exact terms)
  3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF)
  4. Take fused top-N -> optional LLM rerank -> top_k for the prompt

Why RRF and not a weighted score blend: cosine similarity and ts_rank live on
completely different, incomparable scales (0..1 vs unbounded tf-based score).
RRF only looks at *rank position* in each list, so it never needs score
normalization and can't be dominated by one signal's scale. This is the same
fusion NASA-style search engines and most production hybrid-RAG stacks use.

This module assumes:
  - Postgres has the `vector` extension enabled (pgvector).
  - `chunks` table has columns: id, document_id, chunk_index, content,
    page, embedding vector(EMBED_DIM), content_tsv tsvector (GIN indexed).
  - An embeddings API (config.embedding_*) turns text into vectors — no local
    ML runtime, so the Vercel serverless bundle stays small.
"""

from dataclasses import dataclass
from io import BytesIO
import logging
import re
from typing import List, Optional

from openai import OpenAI
from pypdf import PdfReader

from app.config import settings
from app.database import fetch_dense_candidates, fetch_lexical_candidates
from app.schemas import SourceChunk

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)
_HEADING_PATTERN = re.compile(r"^(#{1,6}\s|[0-9]+\.\s|[A-Z][A-Z\s]{4,}$)")

RRF_K = 60          # standard RRF damping constant — de-facto default in IR literature
DENSE_CANDIDATES = 25
LEXICAL_CANDIDATES = 25
FUSED_TOP_N = 12    # goes to reranker
FINAL_TOP_K = None  # falls back to settings.top_k if unset


# ---------------------------------------------------------------------------
# Extraction + structural chunking (fixes the whitespace-collapse bug)
# ---------------------------------------------------------------------------

@dataclass
class RawChunk:
    content: str
    page: int


def extract_pages(file_bytes: bytes) -> List[str]:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    if not any(page.strip() for page in pages):
        raise ValueError("No extractable text was found in the PDF.")
    return pages


def chunk_pages(pages: List[str], target_size: int = 1000, overlap_ratio: float = 0.15) -> List[RawChunk]:
    """
    Chunk on paragraph/heading boundaries instead of a blind whitespace-collapsed
    character slice. Keeps tables and lists from fusing into unreadable runs,
    and records the source page for citation.
    """
    chunks: List[RawChunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", page_text) if p.strip()]

        buffer = ""
        for paragraph in paragraphs:
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= target_size or not buffer:
                buffer = candidate
                continue
            chunks.append(RawChunk(content=buffer, page=page_number))
            overlap_chars = int(len(buffer) * overlap_ratio)
            buffer = buffer[-overlap_chars:] + "\n\n" + paragraph if overlap_chars else paragraph
        if buffer:
            chunks.append(RawChunk(content=buffer, page=page_number))

    return chunks


# ---------------------------------------------------------------------------
# Embeddings (remote API — keeps the serverless bundle free of torch/ONNX)
# ---------------------------------------------------------------------------

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = OpenAI(api_key=settings.embedding_api_key, base_url=settings.embedding_base_url)
    response = client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in response.data]


def embed_query(question: str) -> List[float]:
    return embed_texts([question])[0]


# ---------------------------------------------------------------------------
# Hybrid retrieval: dense + lexical -> RRF fusion -> (optional) rerank
# ---------------------------------------------------------------------------

def _rrf_fuse(*ranked_lists: List[str]) -> List[str]:
    """
    ranked_lists: each is a list of chunk_ids already ordered best-first.
    Returns chunk_ids ordered by fused RRF score, best-first.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for position, chunk_id in enumerate(ranked):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + position + 1)
    return sorted(scores, key=lambda cid: scores[cid], reverse=True)


def hybrid_search(document_id: str, question: str, top_k: Optional[int] = None) -> List[SourceChunk]:
    top_k = top_k or FINAL_TOP_K or settings.top_k

    try:
        query_vector = embed_query(question)
        dense = fetch_dense_candidates(document_id, query_vector, limit=DENSE_CANDIDATES)
    except Exception:
        logger.exception("Dense retrieval failed; continuing on lexical only.")
        dense = []

    try:
        lexical = fetch_lexical_candidates(document_id, question, limit=LEXICAL_CANDIDATES)
    except Exception:
        logger.exception("Lexical retrieval failed; continuing on dense only.")
        lexical = []

    if not dense and not lexical:
        return []

    dense_ids = [row["id"] for row in dense]
    lexical_ids = [row["id"] for row in lexical]
    fused_ids = _rrf_fuse(dense_ids, lexical_ids)[:FUSED_TOP_N]

    by_id = {row["id"]: row for row in (dense + lexical)}
    candidates = [by_id[cid] for cid in fused_ids if cid in by_id]

    reranked = rerank(question, candidates) if settings.rerank_enabled else candidates
    top = reranked[:top_k]

    return [
        SourceChunk(rank=i + 1, score=float(row.get("rerank_score", row.get("score", 0.0))),
                    content=row["content"], page=row.get("page"))
        for i, row in enumerate(top)
    ]


# ---------------------------------------------------------------------------
# Reranking: cheap LLM pass over the fused candidates, precision over recall
# ---------------------------------------------------------------------------

def rerank(question: str, candidates: List[dict]) -> List[dict]:
    """
    Ask the LLM to score each candidate's relevance 0-10. This trades one
    extra cheap call for materially better precision@k than raw RRF order,
    since RRF only reflects retrieval-signal agreement, not actual relevance.
    """
    if not candidates:
        return candidates

    numbered = "\n\n".join(f"[{i}] {c['content'][:600]}" for i, c in enumerate(candidates))
    prompt = (
        "Score how relevant each numbered passage is to the question, 0-10. "
        "Reply ONLY with lines like '0: 7', one per passage, no other text.\n\n"
        f"Question: {question}\n\nPassages:\n{numbered}"
    )

    try:
        client = OpenAI(api_key=settings.answer_api_key, base_url=settings.answer_base_url)
        response = client.chat.completions.create(
            model=settings.answer_model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
        scores = dict(re.findall(r"(\d+)\s*:\s*(\d+(?:\.\d+)?)", raw))
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores.get(str(i), 0))
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    except Exception:
        logger.exception("Reranking failed; falling back to fused RRF order.")
        return candidates


# ---------------------------------------------------------------------------
# Answer generation (unchanged behavior, now with page-aware citations)
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


def _fallback_answer(question: str, sources: List[SourceChunk]) -> str:
    if not sources:
        return "I couldn't find relevant text in the uploaded document for that question."
    question_terms = set(_tokens(question))
    candidates = []
    for source in sources:
        for sentence in re.split(r"(?<=[.!?])\s+", source.content):
            if sentence.strip():
                overlap = len(question_terms & set(_tokens(sentence)))
                candidates.append((overlap, source.rank, sentence.strip()))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    selected = [item[2] for item in candidates[:3]] or [sources[0].content]
    return "Based on the uploaded document:\n\n" + " ".join(selected)


def generate_answer(question: str, sources: List[SourceChunk]) -> str:
    context = "\n\n".join(
        f"Source [{item.rank}] (page {item.page or '?'}):\n{item.content}" for item in sources
    )
    system_prompt = (
        "Answer only using the provided context. Cite sources inline like [1], [2] "
        "matching the source numbers given. If the answer is not in the context, say so plainly."
    )

    for _, api_key, base_url, model in settings.answer_providers:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=20, max_retries=0)
            response = client.chat.completions.create(
                model=model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            answer = (response.choices[0].message.content or "").strip()
            if answer:
                return answer
        except Exception:
            logger.exception("Answer provider failed; trying next provider.")
            continue

    return _fallback_answer(question, sources)