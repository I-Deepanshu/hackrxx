from app.llm_groq import run_llm
import json, re
def explain_and_answer(question: str, evidence_texts: list):
    prompt = f"""You are an expert assistant. Answer only from evidence.
Question: {question}\n\nEvidence:\n"""
    for i,e in enumerate(evidence_texts,1):
        prompt += f"[{i}] (doc:{e.get('doc_id')})\n{e.get('text_snippet')}\n\n"
    prompt += "\nInstructions: Answer succinctly, extract factual fields if present, give short rationale and confidence (0-1). Return JSON with keys: answer, facts, rationale, confidence."
    resp = run_llm(prompt)
    content = resp.get('answer') if isinstance(resp, dict) else str(resp)
    try:
        m = re.search(r"\{.*\}", content, re.S)
        jtext = m.group(0) if m else content
        parsed = json.loads(jtext)
    except Exception:
        parsed = {'answer': content.strip(), 'facts': {}, 'rationale': '', 'confidence': 0.0}
    return parsed
