from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app import models, db
from app.utils.chunking import clean_text  # Import the cleaning function

def create_chunk(db: Session, document_url: str, chunk_text: str, token_count: int):
    try:
        # Clean the text before storing to handle NUL characters
        cleaned_text = clean_text(chunk_text)
        if not cleaned_text:  # Skip empty chunks
            return None
            
        chunk = models.DocumentChunk(
            document_url=document_url, 
            chunk_text=cleaned_text,  # Use cleaned text
            token_count=token_count
        )
        db.add(chunk)
        db.commit()
        db.refresh(chunk)
        return chunk
        
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error creating chunk: {str(e)}")
        return None

def list_chunks(db: Session, limit: int = 100):
    return db.query(models.DocumentChunk)\
           .order_by(models.DocumentChunk.created_at.desc())\
           .limit(limit)\
           .all()
