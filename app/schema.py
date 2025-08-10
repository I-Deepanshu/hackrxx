from pydantic import BaseModel, HttpUrl
from typing import List, Optional

class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class EvidenceItem(BaseModel):
    doc_id: str
    page: Optional[int] = None
    chunk_id: str
    text_snippet: str
    similarity_score: float
    extracted_facts: Optional[dict] = None

class AnswerItem(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[EvidenceItem]
    rationale: str

class RunResponse(BaseModel):
    answers: List[AnswerItem]
