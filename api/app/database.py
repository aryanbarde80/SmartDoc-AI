from contextlib import contextmanager
from typing import Any, Dict, List, Optional
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from app.config import settings

# A single pool reused across warm serverless invocations. Opening a fresh
# TCP+TLS connection per request (the old get_connection()) adds ~100-200ms
# per call and exhausts Postgres connection limits under any real load.
_pool: Optional[ConnectionPool] = None


def _get_pool() -> ConnectionPool:
    global _pool
    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is not configured.")
    if _pool is None:
        _pool = ConnectionPool(
            settings.postgres_url,
            min_size=1,
            max_size=5,
            kwargs={"row_factory": dict_row},
        )
    return _pool


@contextmanager
def get_connection():
    pool = _get_pool()
    with pool.connection() as connection:
        yield connection


def init_db() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_log (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT NOW(),
                    pdf_name TEXT,
                    answer_length INTEGER
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_log_timestamp ON usage_log (timestamp);"
            )

            # Document is now just metadata; the old JSONB chunks blob is gone.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id UUID PRIMARY KEY,
                    pdf_name TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )

            # One row per chunk, with its own dense vector + lexical tsvector.
            cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    page INTEGER,
                    content TEXT NOT NULL,
                    embedding vector({settings.embedding_dim}),
                    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);"
            )
            # HNSW is pgvector's approximate-nearest-neighbor index — fast at
            # query time; ivfflat is the alternative if HNSW isn't available
            # on your Postgres version.
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON chunks USING hnsw (embedding vector_cosine_ops);
                """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (content_tsv);"
            )
        connection.commit()


# ---------------------------------------------------------------------------
# Document + chunk storage
# ---------------------------------------------------------------------------

def store_document_meta(document_id: str, pdf_name: str) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO documents (id, pdf_name)
                VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET pdf_name = EXCLUDED.pdf_name;
                """,
                (document_id, pdf_name),
            )
        connection.commit()


def store_chunks(document_id: str, rows: List[Dict[str, Any]]) -> None:
    """
    rows: [{id, chunk_index, page, content, embedding}, ...]
    Batched with executemany so a 200-chunk document is one round trip family,
    not 200 separate ones.
    """
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunks (id, document_id, chunk_index, page, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                [
                    (row["id"], document_id, row["chunk_index"], row.get("page"),
                     row["content"], row["embedding"])
                    for row in rows
                ],
            )
        connection.commit()


def fetch_document_meta(document_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, pdf_name, created_at FROM documents WHERE id = %s;",
                (document_id,),
            )
            return cursor.fetchone()


def document_has_chunks(document_id: str) -> bool:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM chunks WHERE document_id = %s LIMIT 1;",
                (document_id,),
            )
            return cursor.fetchone() is not None


# ---------------------------------------------------------------------------
# Hybrid retrieval candidate fetches — used by rag.hybrid_search()
# ---------------------------------------------------------------------------

def fetch_dense_candidates(document_id: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
    """Cosine-similarity nearest neighbors, scoped to this document only."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, page, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE document_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (query_vector, document_id, query_vector, limit),
            )
            return cursor.fetchall()


def fetch_lexical_candidates(document_id: str, question: str, limit: int) -> List[Dict[str, Any]]:
    """Postgres full-text search — catches exact terms/numbers embeddings can miss."""
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT id, page, content,
                       ts_rank(content_tsv, websearch_to_tsquery('english', %s)) AS score
                FROM chunks
                WHERE document_id = %s
                  AND content_tsv @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s;
                """,
                (question, document_id, question, limit),
            )
            return cursor.fetchall()


# ---------------------------------------------------------------------------
# Usage logging + insights (unchanged from Phase 1/2)
# ---------------------------------------------------------------------------

def log_question(question: str, pdf_name: Optional[str], answer_length: int) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO usage_log (question, pdf_name, answer_length)
                VALUES (%s, %s, %s);
                """,
                (question, pdf_name, answer_length),
            )
        connection.commit()


def fetch_insights() -> Dict[str, Any]:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total_questions FROM usage_log;")
            total_questions = cursor.fetchone()["total_questions"]

            cursor.execute(
                """
                SELECT question, COUNT(*)::INT AS count
                FROM usage_log
                GROUP BY question
                ORDER BY COUNT(*) DESC, question ASC
                LIMIT 10;
                """
            )
            most_asked_questions = cursor.fetchall()

            cursor.execute(
                """
                SELECT DATE(timestamp) AS day, COUNT(*)::INT AS count
                FROM usage_log
                WHERE timestamp > NOW() - INTERVAL '30 days'
                GROUP BY DATE(timestamp)
                ORDER BY day ASC;
                """
            )
            questions_per_day = cursor.fetchall()

            cursor.execute(
                """
                SELECT pdf_name FROM usage_log
                WHERE pdf_name IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1;
                """
            )
            latest_row = cursor.fetchone()

            return {
                "total_questions": total_questions,
                "most_asked_questions": most_asked_questions,
                "questions_per_day": questions_per_day,
                "latest_pdf_name": latest_row["pdf_name"] if latest_row else None,
            }