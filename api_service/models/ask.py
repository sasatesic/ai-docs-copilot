# api_service/models/ask.py

from typing import List, Optional
from pydantic import BaseModel


class RAGSource(BaseModel):
    score: float
    text: str
    source_file: Optional[str] = None
    chunk_index: Optional[int] = None
    source_id: Optional[str] = None


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[RAGSource]
    used_rag: bool

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    source_id: Optional[str] = None

class SearchHit(BaseModel):
    score: float
    text: str
    source_file: Optional[str] = None
    source_id: Optional[str] = None
    chunk_index: Optional[int] = None

class SearchResponse(BaseModel):
    hits: List[SearchHit]
