import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=api_key)
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def call_gemini(messages: list) -> tuple[str, int]:
    response = client.models.generate_content(
        model=model_name,
        contents=messages,
    )
    text = (response.text or "").strip()
    token_estimate = len(text.split())
    return text, token_estimate

if __name__ == "__main__":
    response, tokens = call_gemini(["Say a short sentence."])
    print(response)
    print(f"Estimated tokens: {tokens}")
