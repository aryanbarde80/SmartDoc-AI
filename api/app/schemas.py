from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    document_id: str
    pdf_name: str
    chunk_count: int


class AskRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    rank: int
    score: float
    content: str
    page: Optional[int] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    pdf_name: str
    timestamp: datetime


class QuestionCount(BaseModel):
    question: str
    count: int


class DailyCount(BaseModel):
    day: date
    count: int


class InsightsResponse(BaseModel):
    total_questions: int
    most_asked_questions: List[QuestionCount]
    questions_per_day: List[DailyCount]
    latest_pdf_name: Optional[str] = None
