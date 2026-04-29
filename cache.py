from sentence_transformers import SentenceTransformer
import numpy as np
import re

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "why",
}
MIN_TEMPLATE_TOKEN_COUNT = 6
MAX_SMALL_EDIT_TOKEN_CHANGES = 4
MIN_TEMPLATE_OVERLAP_RATIO = 0.60
SHORT_QUERY_TOKEN_LIMIT = 6


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def embed(text: str) -> np.ndarray:
    return model.encode(normalize(text), normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def content_tokens(text: str) -> set[str]:
    # Keep apostrophes inside words like "isn't".
    # Also treat "'fett'" and "fett" as the same token.
    words = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", normalize(text))
    return {word for word in words if word not in STOPWORDS}


class SemanticCache:
    def __init__(self, threshold: float = 0.45):
        self.threshold = threshold
        self.entries = []  # list of {embedding, query, response, output_token_estimate}
        self.hits = 0
        self.misses = 0

    def get(self, query: str):
        if not self.entries:
            self.misses += 1
            return None, 0.0, 0

        query_embedding = embed(query)
        scored_entries = []

        for entry in self.entries:
            score = cosine_similarity(query_embedding, entry["embedding"])
            scored_entries.append((score, entry))

        scored_entries.sort(key=lambda item: item[0], reverse=True)
        best_score = scored_entries[0][0]

        for score, entry in scored_entries:
            if score < self.threshold:
                break

            if self._should_reject_response_reuse(
                new_query=query,
                cached_query=entry["query"],
                cached_response=entry["response"],
            ):
                continue

            self.hits += 1
            return entry["response"], score, entry["output_token_estimate"]

        self.misses += 1
        return None, best_score, 0

    def store(self, query: str, response: str, output_token_estimate: int):
        self.entries.append(
            {
                "embedding": embed(query),
                "query": query,
                "response": response,
                "output_token_estimate": output_token_estimate,
            }
        )

    def rollback_miss(self) -> None:
        """Undo the most recent miss when generation fails after lookup."""
        if self.misses > 0:
            self.misses -= 1

    def stats(self):
        total = self.hits + self.misses

        if total > 0:
            hit_rate = (self.hits / total) * 100
        else:
            hit_rate = 0.0

        return {
            "total_queries": total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }

    def _should_reject_response_reuse(
        self,
        new_query: str,
        cached_query: str,
        cached_response: str,
    ) -> bool:
        """Block reuse for risky short swaps and long template-swap cases."""
        old_query_tokens = content_tokens(cached_query)
        new_query_tokens = content_tokens(new_query)
        response_tokens = content_tokens(cached_response)
        min_query_token_count = min(len(old_query_tokens), len(new_query_tokens))

        removed_tokens = old_query_tokens - new_query_tokens
        added_tokens = new_query_tokens - old_query_tokens

        if not removed_tokens or not added_tokens:
            return False

        # Short-query rule: only block direct one-word subject swaps when the
        # removed word still appears in the cached response.
        if min_query_token_count < SHORT_QUERY_TOKEN_LIMIT:
            if (
                len(removed_tokens) == 1
                and len(added_tokens) == 1
                and any(token in response_tokens for token in removed_tokens)
            ):
                return True
            return False

        # Long-query rule: block near-identical sentence templates with small
        # content-word edits (for example, fox -> cat).
        shared_tokens = old_query_tokens & new_query_tokens
        overlap_ratio = (
            len(shared_tokens) / min_query_token_count
            if min_query_token_count
            else 0.0
        )
        total_edit_tokens = len(removed_tokens) + len(added_tokens)

        if (
            min_query_token_count >= MIN_TEMPLATE_TOKEN_COUNT
            and overlap_ratio >= MIN_TEMPLATE_OVERLAP_RATIO
            and total_edit_tokens <= MAX_SMALL_EDIT_TOKEN_CHANGES
        ):
            return True

        return False
