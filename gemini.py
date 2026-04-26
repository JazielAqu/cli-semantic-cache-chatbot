import math
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=api_key)
model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class GeminiRateLimitError(RuntimeError):
    pass


class GeminiRequestError(RuntimeError):
    pass


def call_gemini(messages: list) -> tuple[str, int]:
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=messages,
        )
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            raise GeminiRateLimitError(
                "Gemini rate limit reached for this model."
            ) from exc
        raise GeminiRequestError(f"Gemini request failed: {error_text}") from exc

    text = (response.text or "").strip()
    token_estimate = math.ceil(len(text) / 4)  # rough estimate: 1 token ~ 4 chars
    return text, token_estimate
