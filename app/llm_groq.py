import os
try:
    from groq import Groq
except Exception:
    raise ImportError('Please install groq: pip install groq')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise EnvironmentError('Set GROQ_API_KEY')

client = Groq(api_key=GROQ_API_KEY)
DEFAULT_MODEL = os.getenv('GROQ_MODEL', 'llama3-70b-8192')

def run_llm(prompt: str, max_tokens: int = 512):
    messages = [{'role':'system','content':'You are an evidence-based insurance assistant.'},
                {'role':'user','content': prompt}]
    resp = client.chat.completions.create(model=DEFAULT_MODEL, messages=messages, max_tokens=max_tokens)
    try:
        choice = resp.choices[0]
        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
            return {'answer': choice.message.content, 'raw': resp}
        elif isinstance(choice, dict):
            return {'answer': choice.get('message', {}).get('content') or choice.get('text'), 'raw': resp}
        else:
            return {'answer': str(choice), 'raw': resp}
    except Exception:
        return {'answer': str(resp), 'raw': resp}
