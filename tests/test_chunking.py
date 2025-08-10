from app.utils.chunking import chunk_text_token_aware
def test_chunking_short_text():
    text = 'This is a short document.'
    chunks = chunk_text_token_aware(text, model_name='gpt-3.5-turbo', chunk_size=50)
    assert len(chunks) >= 1
    assert 'text' in chunks[0]
