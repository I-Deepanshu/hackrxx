from sqlalchemy.orm import Session
from app import models, db

def create_chunk(db: Session, document_url: str, chunk_text: str, token_count: int):
    chunk = models.DocumentChunk(document_url=document_url, chunk_text=chunk_text, token_count=token_count)
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk

def list_chunks(db: Session, limit: int = 100):
    return db.query(models.DocumentChunk).order_by(models.DocumentChunk.created_at.desc()).limit(limit).all()
