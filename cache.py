from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def embed(text: str) -> np.ndarray:
    return model.encode(normalize(text), normalize_embeddings=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


class SemanticCache:
    def __init__(self, threshold: float = 0.66):
        self.threshold = threshold
        self.entries = []  # list of {embedding, query, response, output_token_estimate}
        self.hits = 0
        self.misses = 0

    def get(self, query: str):
        if not self.entries:
            self.misses += 1
            return None, 0.0, 0

        query_embedding = embed(query)
        best_score = float("-inf")
        best_entry = None

        for entry in self.entries:
            score = cosine_similarity(query_embedding, entry["embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            self.hits += 1
            return best_entry["response"], best_score, best_entry["output_token_estimate"]

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

    def stats(self):
        total = self.hits + self.misses

        if total > 0:
            hit_rate = ((self.hits / total) * 100) 
        else:
            hit_rate = 0.0

        return {
            "total_queries": total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }
