from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import (
    document_has_chunks,
    fetch_document_meta,
    fetch_insights,
    init_db,
    log_question,
    store_chunks,
    store_document_meta,
)
from app.rag import chunk_pages, embed_texts, extract_pages, generate_answer, hybrid_search
from app.schemas import AskRequest, AskResponse, InsightsResponse, UploadResponse

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # server-side cap — the old "20 MB" was UI copy only


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()
    except Exception:
        logger.exception("Database initialization failed; database-backed actions may be unavailable.")
    yield


app = FastAPI(title="SmartDoc AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def healthcheck():
    return {
        "status": "ok",
        "postgres_configured": bool(settings.postgres_url),
        "answer_provider": settings.answer_provider,
        "answer_api_configured": bool(settings.answer_api_key),
        "backup_provider_configured": bool(settings.groq_api_key),
        "embeddings_configured": settings.embeddings_configured,
    }


@app.post("/api/upload", response_model=UploadResponse)
def upload_pdf(file: UploadFile = File(...)):
    # Sync def, not async def: psycopg and the OpenAI client here are both
    # blocking calls. FastAPI runs sync endpoints in a threadpool automatically,
    # so this no longer blocks the event loop for the whole embedding/DB round trip.
    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="The uploaded PDF is empty.")
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="PDF exceeds the 20 MB limit.")

    if not settings.embeddings_configured:
        raise HTTPException(
            status_code=500,
            detail="Embedding provider is not configured (EMBEDDING_API_KEY missing).",
        )

    try:
        pages = extract_pages(file_bytes)
        chunks = chunk_pages(pages)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to process the PDF.") from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks were created from the PDF.")

    try:
        vectors = embed_texts([c.content for c in chunks])
    except Exception as exc:
        # Fail loudly rather than storing a document that will silently give
        # worse (lexical-only) answers for its entire lifetime.
        logger.exception("Embedding failed during upload.")
        raise HTTPException(
            status_code=502,
            detail="Could not generate embeddings for this document. Please try again.",
        ) from exc

    document_id = str(uuid4())
    try:
        store_document_meta(document_id, filename)
        store_chunks(
            document_id,
            [
                {
                    "id": str(uuid4()),
                    "chunk_index": i,
                    "page": chunk.page,
                    "content": chunk.content,
                    "embedding": vector,
                }
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ],
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to store document/chunks.")
        raise HTTPException(status_code=500, detail="Failed to store the processed document.") from exc

    return UploadResponse(document_id=document_id, pdf_name=filename, chunk_count=len(chunks))


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    try:
        meta = fetch_document_meta(payload.document_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load document %s.", payload.document_id)
        raise HTTPException(status_code=503, detail="The document database is temporarily unavailable.") from exc

    if not meta:
        raise HTTPException(status_code=404, detail="Document not found. Upload the PDF again.")

    pdf_name = meta["pdf_name"]

    try:
        if not document_has_chunks(payload.document_id):
            raise HTTPException(status_code=404, detail="This document has no indexed content. Re-upload it.")

        sources = hybrid_search(payload.document_id, payload.question, top_k=settings.top_k)
        answer = generate_answer(payload.question, sources)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to answer question for document %s.", payload.document_id)
        raise HTTPException(status_code=500, detail="Failed to answer the question.") from exc

    try:
        log_question(payload.question, pdf_name, len(answer))
    except Exception:
        logger.exception("Could not log question analytics.")

    return AskResponse(answer=answer, sources=sources, pdf_name=pdf_name, timestamp=datetime.now(timezone.utc))


@app.get("/api/insights", response_model=InsightsResponse)
def insights():
    try:
        data = fetch_insights()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load insights.")
        raise HTTPException(status_code=500, detail="Failed to load insights.") from exc

    return InsightsResponse(**data)