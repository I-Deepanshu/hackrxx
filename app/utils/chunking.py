from typing import List, Dict
import re

def clean_text(text: str) -> str:
    # Remove null characters and other problematic characters
    text = text.replace('\x00', '')
    # Clean other special characters but preserve newlines
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

def chunk_text_token_aware(text: str, max_chunk_length: int = 700) -> List[Dict]:
    # Clean text before chunking
    cleaned_text = clean_text(text)
    chunks = []
    current_chunk = ''
    current_length = 0
    
    # Split by sentences while respecting paragraph breaks
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_text)
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_chunk_length:
            if current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': f'chunk_{len(chunks)}',
                    'token_count': current_length
                })
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk = f'{current_chunk} {sentence}' if current_chunk else sentence
            current_length += sentence_length
    
    if current_chunk:
        chunks.append({
            'text': current_chunk.strip(),
            'chunk_id': f'chunk_{len(chunks)}',
            'token_count': current_length
        })
    
    return chunks
