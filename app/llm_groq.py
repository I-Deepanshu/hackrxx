import os
from typing import Dict, Any, Optional
try:
    from groq import Groq
except Exception:
    raise ImportError('Please install groq: pip install groq')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise EnvironmentError('Set GROQ_API_KEY')

DEFAULT_MODEL = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
DEFAULT_MAX_TOKENS = int(os.getenv('GROQ_MAX_TOKENS', '512'))
DEFAULT_TEMPERATURE = float(os.getenv('GROQ_TEMPERATURE', '0.1'))
DEFAULT_TOP_P = float(os.getenv('GROQ_TOP_P', '0.9'))

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a precise insurance policy analyzer specialized in providing detailed, structured responses about policy terms and conditions. Follow these exact guidelines:

1. Response Structure:
   - Begin with a clear, direct answer
   - Include specific numerical values (days, percentages, years)
   - State all conditions and eligibility criteria
   - Mention any limitations or caps

2. Key Information Formatting:
   - Waiting Periods: Always specify in months/years (e.g., "thirty-six (36) months")
   - Monetary Values: Use both percentage and reference (e.g., "1% of Sum Insured")
   - Time Periods: Use full words for numbers under hundred (e.g., "thirty days")
   - Conditions: List all qualifying criteria

3. Critical Elements to Include:
   - Coverage Limits and Sub-limits
   - Eligibility Requirements
   - Renewal Conditions
   - Applicable Time Periods
   - Facility/Institution Requirements
   - Regulatory Compliance References

4. Response Style:
   - Use complete, well-structured sentences
   - Include all relevant qualifiers and conditions
   - Maintain formal, precise language
   - Connect related conditions with appropriate conjunctions
   - Specify when benefits are subject to specific conditions

5. Accuracy Requirements:
   - Quote exact figures from the policy
   - Include regulatory references when mentioned
   - Specify institutional requirements precisely
   - Detail all applicable limits and caps

Remember: Each answer should be self-contained and comprehensive, providing all relevant information from the policy document without requiring additional context."""

def run_llm(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    system_prompt: str = SYSTEM_PROMPT
) -> Dict[str, Any]:
    """
    Run the LLM with specified parameters.
    
    Parameters:
        prompt (str): The user prompt
        max_tokens (int): Maximum tokens in response (default: 512)
        temperature (float): Controls randomness (0.1 for precise responses)
        top_p (float): Nucleus sampling parameter (0.9 for balanced responses)
        system_prompt (str): Detailed system behavior definition
    
    Model Capabilities (llama3-70b-8192):
        - Context Window: 8,192 tokens
        - Optimal Response Length: 100-800 tokens for insurance queries
        - Temperature: Keep at 0.1 for consistent policy information
        - Top_p: Maintain at 0.9 for accurate yet natural responses
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    
    try:
        resp = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        
        if resp.choices:
            choice = resp.choices[0]
            if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                return {
                    'answer': choice.message.content,
                    'raw': resp,
                    'metadata': {
                        'model': DEFAULT_MODEL,
                        'max_tokens': max_tokens,
                        'temperature': temperature,
                        'top_p': top_p
                    }
                }
            elif isinstance(choice, dict):
                return {
                    'answer': choice.get('message', {}).get('content') or choice.get('text'),
                    'raw': resp,
                    'metadata': {
                        'model': DEFAULT_MODEL,
                        'max_tokens': max_tokens,
                        'temperature': temperature,
                        'top_p': top_p
                    }
                }
        
        return {
            'answer': 'Unable to generate response',
            'raw': resp,
            'error': 'Invalid response format'
        }
        
    except Exception as e:
        return {
            'answer': f'Error generating response: {str(e)}',
            'raw': None,
            'error': str(e)
        }
