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


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 0
    return math.ceil(len(text) / 4)  # rough estimate: 1 token ~ 4 chars


def estimate_input_tokens(messages: list[str]) -> int:
    prompt_text = "\n".join(messages)
    return estimate_tokens_from_text(prompt_text)


def call_gemini(messages: list[str]) -> tuple[str, int, int]:
    input_token_estimate = estimate_input_tokens(messages)

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
    output_token_estimate = estimate_tokens_from_text(text)
    return text, input_token_estimate, output_token_estimate
