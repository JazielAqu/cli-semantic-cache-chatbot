from urllib import response

from cache import SemanticCache
from gemini import call_gemini


cache = SemanticCache(threshold=0.85)
conversation_history: list[dict[str, str]] = []


def format_history_for_gemini() -> list[str]:
    lines: list[str] = []
    for turn in conversation_history:
        lines.append(f"{turn['role'].upper()}: {turn['content']}")

    prompt = "Continue this conversation naturally:\n\n" + "\n".join(lines)
    return [prompt]


def chat(user_input: str) -> tuple[str, bool, float]:
    cached_response, score = cache.get(user_input)

    if cached_response is not None:
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": cached_response})
        return cached_response, True, score

    conversation_history.append({"role": "user", "content": user_input})
    messages = format_history_for_gemini()
    response, token_estimate = call_gemini(messages)
    conversation_history.append({"role": "assistant", "content": response})
    cache.store(user_input, response, token_estimate)
    return response, False, score



def print_stats() -> None:
    stats = cache.stats()
    print("\n--- Session Stats ---")
    print(f"Total lookups: {stats['total_queries']}")
    print(f"Cache hits: {stats['cache_hits']}")
    print(f"Cache misses: {stats['cache_misses']}")
    print(f"Cache hit rate: {stats['hit_rate']}")
    print(f"Estimated tokens saved: {stats['tokens_saved']}")
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
        if lower in {"/exit", "exit", "quit"}:
            print_stats()
            break
        if lower in {"/stats", "stats"}:
            print_stats()
            continue

        response, from_cache, score = chat(user_input)
    
        if from_cache:
            print(f"Bot [cache score={score:.3f}]: {response}\n")
        else:
            print(f"Bot [generated best_score={score:.3f}]: {response}\n")

if __name__ == "__main__":
    main()