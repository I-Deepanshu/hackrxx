from pydantic import BaseModel
from typing import List, Optional

class EvidenceItem(BaseModel):
    doc_id: str
    page: Optional[int]
    chunk_id: str
    text_snippet: str
    similarity_score: float
    extracted_facts: Optional[List[str]]

class AnswerItem(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[EvidenceItem]
    rationale: str

class RunRequest(BaseModel):
    documents: str
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]
