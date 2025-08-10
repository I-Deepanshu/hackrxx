import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
client = TestClient(app)

@patch('app.llm_groq.run_llm')
def test_reasoner_endpoint(mock_run):
    mock_run.return_value = {'answer':'{\"answer\": \"Not stated\", \"facts\": {}, \"rationale\": \"no\", \"confidence\": 0.0}'}
    headers = {'Authorization': 'Bearer d6191acd6bd9dc9d09259ea7e6ee110688affcf1d572b6eda9f54c61890f1dca'}
    data = {'documents': 'https://example.com/doc.txt', 'questions': ['Is knee surgery covered?']}
    r = client.post('/api/v1/hackrx/run', json=data, headers=headers)
    assert r.status_code == 200
    assert 'answers' in r.json()
