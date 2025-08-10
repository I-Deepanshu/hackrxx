from app.embeddings_ import get_embedding
try:
    from app.pinecone_client import get_index
except Exception:
    get_index = None

def upsert_chunks(doc_id: str, chunks):
    if get_index is None:
        return
    idx = get_index()
    vectors = []
    for c in chunks:
        emb = get_embedding(c['text'])
        meta = {'doc_id': doc_id, 'chunk_id': c['chunk_id']}
        vectors.append((f"{doc_id}::{c['chunk_id']}", emb, meta))
    idx.upsert(vectors=vectors)

def query_top_k(query_text: str, k: int =5):
    if get_index is None:
        return []
    idx = get_index()
    q_emb = get_embedding(query_text)
    resp = idx.query(vector=q_emb, top_k=k, include_metadata=True)
    results = []
    for match in resp.get('matches', []):
        meta = match.get('metadata', {})
        score = match.get('score', match.get('distance', 0))
        results.append({'chunk_id': meta.get('chunk_id'), 'doc_id': meta.get('doc_id'), 'score': score})
    return results
