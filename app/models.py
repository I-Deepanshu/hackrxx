import uuid
from sqlalchemy import Column, String, Integer, DateTime, LargeBinary, Text, func
from sqlalchemy.dialects.postgresql import UUID
from app.db import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_url = Column(String, nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
