from cache import SemanticCache
from gemini import (
    GeminiRateLimitError,
    GeminiRequestError,
    call_gemini,
    estimate_input_tokens,
)


cache = SemanticCache(threshold=0.45)
conversation_history: list[dict[str, str]] = []
token_metrics = {
    "estimated_input_tokens_sent": 0,
    "estimated_output_tokens_generated": 0,
    "estimated_input_tokens_saved": 0,
    "estimated_output_tokens_saved": 0,
}
runtime_metrics = {
    "generated_responses": 0,
    "cache_responses": 0,
    "llm_errors": 0,
}


def format_history_for_gemini() -> list[str]:
    lines: list[str] = []
    for turn in conversation_history:
        lines.append(f"{turn['role'].upper()}: {turn['content']}")

    prompt = "Continue this conversation naturally:\n\n" + "\n".join(lines)
    return [prompt]


def chat(user_input: str) -> tuple[str, bool, float]:
    cached_response, score, cached_output_tokens = cache.get(user_input)

    if cached_response is not None:
        conversation_history.append({"role": "user", "content": user_input})
        would_have_sent_messages = format_history_for_gemini()
        token_metrics["estimated_input_tokens_saved"] += estimate_input_tokens(
            would_have_sent_messages
        )
        token_metrics["estimated_output_tokens_saved"] += cached_output_tokens
        conversation_history.append({"role": "assistant", "content": cached_response})
        return cached_response, True, score

    conversation_history.append({"role": "user", "content": user_input})
    messages = format_history_for_gemini()
    try:
        response, input_tokens_estimate, output_tokens_estimate = call_gemini(messages)
    except (GeminiRateLimitError, GeminiRequestError):
        # Remove unresolved user turn when generation fails.
        conversation_history.pop()
        # Lookup was a miss, but no response was produced, so do not count it.
        cache.rollback_miss()
        raise

    token_metrics["estimated_input_tokens_sent"] += input_tokens_estimate
    token_metrics["estimated_output_tokens_generated"] += output_tokens_estimate
    conversation_history.append({"role": "assistant", "content": response})
    cache.store(user_input, response, output_tokens_estimate)
    return response, False, score



def print_stats() -> None:
    stats = cache.stats()
    successful_answered_turns = (
        runtime_metrics["generated_responses"] + runtime_metrics["cache_responses"]
    )
    total_estimated_saved = (
        token_metrics["estimated_input_tokens_saved"]
        + token_metrics["estimated_output_tokens_saved"]
    )
    total_estimated_sent_and_generated = (
        token_metrics["estimated_input_tokens_sent"]
        + token_metrics["estimated_output_tokens_generated"]
    )

    print("\n--- Session Stats ---")
    print(f"Total lookups: {stats['total_queries']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Cache hit rate: {stats['hit_rate']}")
    print(f"Successful answered turns: {successful_answered_turns}")
    print(f"LLM error turns: {runtime_metrics['llm_errors']}")
    print(f"Estimated input tokens sent: {token_metrics['estimated_input_tokens_sent']}")
    print(
        f"Estimated output tokens generated: "
        f"{token_metrics['estimated_output_tokens_generated']}"
    )
    print(
        f"Estimated input tokens saved via cache: "
        f"{token_metrics['estimated_input_tokens_saved']}"
    )
    print(
        f"Estimated output tokens saved via cache: "
        f"{token_metrics['estimated_output_tokens_saved']}"
    )
    print(f"Estimated total tokens saved: {total_estimated_saved}")
    print(f"Estimated total tokens sent/generated: {total_estimated_sent_and_generated}")
    print("---------------------\n")


def main() -> None:
    print("Chatbot ready. Type '/exit' to quit or '/stats' for cache stats.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print_stats()
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower == "/exit":
            print_stats()
            break
        if lower == "/stats":
            print_stats()
            continue

        try:
            response, from_cache, score = chat(user_input)
        except GeminiRateLimitError as exc:
            runtime_metrics["llm_errors"] += 1
            print(f"Bot [error]: {exc}\n")
            continue
        except GeminiRequestError as exc:
            runtime_metrics["llm_errors"] += 1
            print(f"Bot [error]: {exc}\n")
            continue
    
        if from_cache:
            runtime_metrics["cache_responses"] += 1
            print(f"Bot [cache score={score:.3f}]: {response}\n")
        else:
            runtime_metrics["generated_responses"] += 1
            print(f"Bot [generated best_score={score:.3f}]: {response}\n")

if __name__ == "__main__":
    main()
