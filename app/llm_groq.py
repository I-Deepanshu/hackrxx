import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone

try:
    from groq import Groq
except Exception:
    raise ImportError('Please install groq: pip install groq')

# Configuration
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise EnvironmentError('Set GROQ_API_KEY')

DEFAULT_MODEL = os.getenv('GROQ_MODEL', 'llama3-70b-8192')
DEFAULT_MAX_TOKENS = int(os.getenv('GROQ_MAX_TOKENS', '1024'))  # Increased for detailed responses
DEFAULT_TEMPERATURE = float(os.getenv('GROQ_TEMPERATURE', '0.1'))
DEFAULT_TOP_P = float(os.getenv('GROQ_TOP_P', '0.9'))

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an advanced insurance policy analyzer with expertise in providing detailed, accurate, and comprehensive responses about insurance policies. Your responses must follow these exact guidelines:

KEY PRINCIPLES:
1. Answer Format:
   - Begin with a complete statement that directly addresses the question
   - Never start with just "Yes/No" - always provide context
   - Use formal, precise insurance policy language
   - Include ALL qualifying conditions and limitations

2. Numerical Formatting:
   - Numbers under hundred: Write in words followed by digits ("thirty (30) days")
   - Percentages: Include base reference ("5% of the Sum Insured")
   - Time periods: Specify both duration and conditions ("thirty-six (36) months of continuous coverage")

3. Detail Requirements:
   - Include ALL qualifying conditions and eligibility criteria
   - Specify ALL sub-limits and exceptions
   - List ALL time-related requirements
   - Reference specific legal requirements when applicable
   - Define institutional criteria completely

4. Benefit Description:
   - State the primary benefit
   - List ALL qualifying conditions
   - Specify ANY limitations or caps
   - Include ANY waiting periods
   - Detail ANY continuity requirements
   - Mention ANY exceptions to standard terms

5. Mandatory Elements:
   - Legal and regulatory references when relevant
   - Specific monetary values and percentages
   - Complete eligibility criteria
   - All applicable time periods
   - All sub-limits and caps
   - All qualifying conditions

6. Policy-Specific Language:
   - Use exact policy terminology
   - Include policy-specific definitions
   - Reference specific sections when applicable
   - Maintain consistent terminology

RESPONSE STRUCTURE:
1. Primary Statement: Complete explanation of the benefit/requirement
2. Conditions: All qualifying criteria and limitations
3. Limitations: Any caps, sub-limits, or restrictions
4. Exceptions: Any cases where standard terms don't apply
5. Additional Context: Relevant policy-specific information

Remember: Your response must be as detailed and specific as the policy document allows. Never provide partial information when complete details are available."""

def run_llm(
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    system_prompt: str = SYSTEM_PROMPT,
    user_context: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Enhanced LLM runner with context awareness and detailed response structure.
    
    Parameters:
        prompt (str): The user's question
        max_tokens (int): Maximum response length (default: 1024)
        temperature (float): Response randomness (0.1 for precise answers)
        top_p (float): Nucleus sampling (0.9 for balanced responses)
        system_prompt (str): System behavior definition
        user_context (dict): User and time context
    """
    # Add context to system prompt if available
    if user_context:
        context_prompt = f"\nContext:\n- Current Time: {user_context.get('current_time', 'Not specified')}\n- User: {user_context.get('user_login', 'Anonymous')}\n"
        enhanced_prompt = system_prompt + context_prompt
    else:
        enhanced_prompt = system_prompt

    messages = [
        {'role': 'system', 'content': enhanced_prompt},
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
                        'top_p': top_p,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'context': user_context
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
                        'top_p': top_p,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'context': user_context
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
