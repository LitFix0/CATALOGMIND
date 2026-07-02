# test_llm_client.py — throwaway, not part of the pytest suite
from dotenv import load_dotenv
load_dotenv()

from agent.llm_client import call_groq

result = call_groq(
    system_prompt="You are a helpful assistant. Reply in one short sentence.",
    user_prompt="Say hello and confirm you're working.",
)
print("Result:", result)