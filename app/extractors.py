import requests, io
def fetch_blob_text(url: str):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    content_type = r.headers.get('Content-Type','').lower()
    text = r.text
    pages = [{'page': None, 'text': text}]
    return text, pages
