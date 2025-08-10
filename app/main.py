from fastapi import FastAPI, Depends, HTTPException, Header, status
from app.schema import RunRequest, RunResponse, AnswerItem, EvidenceItem, RunResponse
from app.extractors import fetch_blob_text
from app.utils.chunking import chunk_text_token_aware
from app.retriever import upsert_chunks, query_top_k
from app.reasoner import explain_and_answer
from app.db import SessionLocal, engine, Base
from app import models
from app.config import settings
import uuid

Base.metadata.create_all(bind=engine)

app = FastAPI(title='HackRx Retrieval API (Groq + Postgres)')

def verify_token(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing Authorization header')
    if not authorization.startswith('Bearer '):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail='Invalid Authorization header')
    token = authorization.split(' ',1)[1].strip()
    if token != settings.HACKRX_TEAM_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail='Invalid token')
    return True

@app.post('/hackrx/run', response_model=RunResponse)
async def hackrx_run(req: RunRequest, ok: bool = Depends(verify_token)):
    doc_url = req.documents
    raw_text, pages = fetch_blob_text(doc_url)
    # combine pages into text
    full_text = "\n".join([p.get('text','') for p in pages])
    doc_id = str(uuid.uuid4())[:8]
    chunks = chunk_text_token_aware(full_text)
    
    # persist chunks to DB
    db = SessionLocal()
    from app.crud import create_chunk
    for c in chunks:
        try:
            create_chunk(db, document_url=doc_url, chunk_text=c['text'], token_count=c['token_count'])
        except Exception:
            pass
    
    try:
        upsert_chunks(doc_id, chunks)
    except Exception:
        pass
    
    # Initialize answers list for simplified response
    simple_answers = []
    
    # Process each question
    for q in req.questions:
        top = query_top_k(q, k=5)
        evidence = []
        for t in top:
            matching = next((c for c in chunks if c['chunk_id'] == t.get('chunk_id')), None)
            snippet = matching['text'] if matching else ''
            evidence.append({
                'doc_id': t.get('doc_id'),
                'chunk_id': t.get('chunk_id'),
                'text_snippet': snippet,
                'similarity_score': float(t.get('score',0.0))
            })
        
        # Get the parsed response
        parsed = explain_and_answer(q, evidence)
        
        # Only add the answer string to the simple_answers list
        simple_answers.append(parsed.get('answer', 'Not found'))
    
    # Return simplified response format
    return RunResponse(answers=simple_answers)
