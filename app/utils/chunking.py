import tiktoken
from typing import List, Dict
from app.config import settings

def chunk_text_token_aware(text: str, model_name: str = "gpt-4o-mini", chunk_size: int = None, overlap: int = 50):
    "+""Return list of chunks (dicts) with token counts using tiktoken."+""
    if chunk_size is None:
        chunk_size = settings.MAX_CHUNK_TOKENS
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(text)
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append({"chunk_id": f"c_{idx}", "text": chunk_text, "token_count": len(chunk_tokens)})
        idx += 1
        start += chunk_size - overlap
    return chunks
