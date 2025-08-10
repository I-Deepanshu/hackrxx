# Placeholder embedding stub - replace with real embeddings (OpenAI or local)
def get_embedding(text: str):
    return [abs(hash(text)) % 1000] * 1536
