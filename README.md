# CLI Semantic Cache Chatbot 

This project is a Python CLI chatbot with a semantic cache layer.  
The cache reduces unnecessary Gemini calls by reusing a previous response when a new query is semantically similar to a cached query.

## What It Does

- Chat in the terminal (`main.py`).
- Maintains conversation history within a session.
- Checks semantic similarity before each LLM call.
- Returns cached response on hit, generates fresh response on miss.
- Shows clear source labels:
  - `[cache score=...]`
  - `[generated ...]`
- Tracks session metrics:
  - Cache hit/miss counts and hit rate
  - Estimated input/output tokens sent and saved

## Tech Stack

- LLM: Gemini API (`google-genai`)
- Embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
- Similarity: cosine similarity on normalized embeddings
- Language: Python

## Project Files

- `main.py`: CLI loop, command handling, history/context, token metrics, orchestration.
- `cache.py`: semantic embedding, similarity lookup, threshold hit/miss logic, cache stats.
- `gemini.py`: Gemini API call wrapper, token estimation helpers, request/rate-limit error handling.
- `threshold_eval.py`: labeled-pair threshold sweep and evaluation metrics (`precision`, `recall`, `f1`, `accuracy`).

## Setup

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and set your key:

```env
GEMINI_API_KEY=your_real_key_here
```

Optional:

```env
GEMINI_MODEL=gemini-2.5-flash
```

## Run

```bash
python main.py
```

Available commands:

- `/stats`: print current cache + token metrics
- `/exit`: exit and print final stats

## How the Semantic Cache Works

1. Normalize user query (`strip + lowercase + collapse spaces`).
2. Embed normalized query with `all-MiniLM-L6-v2`.
3. Compare against cached query embeddings via cosine similarity.
4. Find best matching cached entry.
5. Apply threshold check (`best_score >= threshold`).
6. Apply guardrails:
   - For very short queries (fewer than 3 content tokens), skip wording-leak guardrails. In testing, applying those rules to short text over-blocked valid cache hits and increased unnecessary Gemini calls.
   - For longer queries, reject cache reuse when the new query changes key words and the cached response still contains removed old words (for example, old query had `fox`, new query has `cat`, and cached response still says `fox`).
7. If both checks pass, return cached response (`cache hit`); otherwise call Gemini (`cache miss`), return generated response, and cache it.

Cache entry fields:

- `embedding`
- `query`
- `response`
- `output_token_estimate`

## Similarity Method and Threshold Choice

I tuned the threshold using `threshold_eval.py`:

- **Initial small-set tuning:** started with a smaller labeled set (about 12 pairs total, 6 paraphrase + 6 non-paraphrase). On that set, `0.43` was the highest threshold with perfect `f1` and zero false positives/false negatives.
- **Expanded offline tuning:** expanded to a larger labeled set with harder negatives (negation traps, lexical overlap traps, typo correction, casual phrasing) and swept thresholds using confusion-matrix metrics (`TP`, `FP`, `TN`, `FN`) plus `precision`, `recall`, `f1`, and `accuracy`.
- **Expanded-set result:** `0.35` gave the best `f1` and no false negatives on that dataset (better for limiting unnecessary Gemini calls).
- **Precision-focused alternative:** `0.45` is also defensible if prioritizing fewer false positives and higher response quality over recall.
- **Production choice:** I set `0.45` in code as the final threshold because it is directly backed by the threshold-eval sweep and is more conservative than the F1-optimal `0.35`, reducing false-positive cache hits at the cost of lower recall.
- **Guardrail pairing:** in addition to the threshold, I added a response-reuse guardrail that rejects high-similarity hits when key query words changed and reuse would likely carry stale wording.

Current runtime threshold in `cache.py` and `main.py` is `0.45`.

## Metrics and Token Estimation

This project reports estimated token usage and savings:

- Estimated input tokens sent on generated calls.
- Estimated output tokens generated on generated calls.
- Estimated input tokens saved on cache hits (full prompt/history that was not sent).
- Estimated output tokens saved on cache hits (cached response generation avoided).

Token estimation uses a simple heuristic:

- `estimated_tokens = ceil(char_count / 4)`

This is intentionally approximate and is labeled as estimated in stats.

## Trade-offs

- **Precision vs recall (with product impact)**: in a conversational UX, a false-positive cache hit (wrong reused answer) is usually worse than a cache miss (extra LLM call). That asymmetry is why I run `0.45` in production instead of the F1-optimal `0.35`.
- **Single-turn cache matching**: cache lookup compares queries in isolation, not full conversation state. The same wording can require different answers in different moments of a conversation, but this cache treats them as equivalent.
- **In-memory cache**: simple and fast for this scope, but not persistent across program restarts.
- **Approximate token accounting**: practical for quick analysis, but not guaranteed to exactly match provider-side usage accounting.
- **Context-sensitive wording artifacts**: semantic similarity can still reuse responses with phrasing tied too closely to the original query wording.
- **Guardrails favor longer queries**: wording-leak checks work better on longer text. On short text, they can over-trigger and cause unnecessary Gemini calls, so short-query behavior relies more on embedding similarity plus threshold.
- **Cross-intent short-query separation is statistical**: in testing, cross-intent short queries were usually misses due to lower similarity, but this is not an explicit rule.

## Failure Modes Observed

- Query pairs with similar meaning but different response tone/context can still produce awkward reuse if threshold is too low.
- Typo-correction flows can trigger valid semantic hits while reusing wording that references the original typo. Example observed during testing: the response to `"How do I go from point A to point B on fett?"` included a typo-specific remark (`"Ah, 'fett'..."`), and when the user corrected the query to `"How do I go from point A to point B on feet?"`, the similarity score was `0.657`.
- Mitigation implemented: a response-reuse guardrail now blocks this typo-leak pattern (and similar small word-substitution template cases like `fox -> cat`) even when similarity is high.
- Mitigation implemented for very short queries: skip wording-leak guardrails under 3 content tokens. This restored valid short-query cache hits (for example greeting variants) and reduced unnecessary Gemini calls.

## Improvements With More Time

- Add an optional SQLite persistence mode for crash recovery/session resume while keeping in-memory mode as the default.
- Scope persisted cache entries by `session_id` (and optionally `user_id`) so cache reuse does not bleed across unrelated conversations.
- Add CLI controls (for example `/new` and `/resume <session_id>`) so users can explicitly choose whether to start fresh or continue.
- Expand the response-reuse guardrail beyond typo/copy-edit cases (for example broader contradiction checks and stronger query-vs-response consistency checks).
- Add larger labeled threshold datasets tailored to the target use case (for example: returns, refunds, order tracking, shipping, billing, and account access support queries).
- Benchmark multiple embedding models (including customer-support or financial-domain options) on the same labeled dataset, then recalibrate thresholds per model. Similarity thresholds are model-dependent, and a better model may reduce typo and word-substitution false positives.
- Add automated tests for cache hit/miss behavior and token accounting.
- Add optional model/backend abstraction for embedding and storage experiments.

## AI Usage Note

I used AI tools as an engineering assistant for brainstorming, code review-style feedback, and writing clarity.  
I owned implementation and final decisions: architecture, threshold strategy, guardrail design, error-handling behavior, metric definitions, and local validation/testing.

## References

### Primary Sources Used

- Gemini API caching docs: https://ai.google.dev/gemini-api/docs/caching
- Sentence Transformers docs: https://sbert.net/
- Accuracy / precision / recall primer: https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall

### Background Reading

- Token concepts: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
- Similarity threshold tuning: https://milvus.io/ai-quick-reference/how-do-you-tune-similarity-thresholds-for-better-relevance
- Cosine threshold Q&A: https://ai.stackexchange.com/questions/40597/how-do-i-choose-a-good-treshold-for-classification-using-cosine-similarity-scor
- Semantic cache article: https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/semantic-cache
- Chat history token cost discussion: https://stackoverflow.com/questions/78743420/does-the-history-of-a-chat-count-towards-the-input-tokens-cost-for-genkit
- Additional background:
  - https://www.e6data.com/blog/embedding-essentials-cosine-similarity-sql-with-vectors
  - https://medium.com/@amitchaudhary_86/from-keywords-to-meaning-a-hands-on-tutorial-with-sentence-transformers-for-semantic-search-c6adf00b9e19
  - https://codesignal.com/learn/courses/behavioral-benchmarking-of-llms/lessons/measuring-and-interpreting-token-usage-in-llms
