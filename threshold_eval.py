from sentence_transformers import SentenceTransformer
import numpy as np

labeled_pairs = [
    ("What's your return policy?", "How do I return something I bought?", 1),
    ("Can I get a refund?", "How do refunds work?", 1),
    ("Where is my order?", "How can I track my shipment?", 1),
    ("I need to change my password", "How do I reset my password?", 1),
    ("Do you offer free shipping?", "Is shipping free?", 1),
    ("How long does delivery take?", "What's your typical shipping time?", 1),

    ("What's your return policy?", "How do I reset my password?", 0),
    ("Where is my order?", "Do you have a student discount?", 0),
    ("How do refunds work?", "How do I delete my account?", 0),
    ("Can I change my shipping address?", "What payment methods do you accept?", 0),
    ("Do you ship internationally?", "What are your business hours?", 0),
    ("How can I contact support?", "How do I apply a promo code?", 0),
]

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

unique_texts = sorted({text for left_text, right_text, _ in labeled_pairs for text in (left_text, right_text)}) 
embeddings = embedding_model.encode(unique_texts, normalize_embeddings=True)
text_to_index = {text: i for i, text in enumerate(unique_texts)}

scored_pairs = []
for left_text, right_text, is_paraphrase in labeled_pairs:
    similarity_score = float(np.dot(embeddings[text_to_index[left_text]], embeddings[text_to_index[right_text]]))
    scored_pairs.append((similarity_score, is_paraphrase, left_text, right_text))


def compute_metrics_at_threshold(threshold: float):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for similarity_score, is_paraphrase, _, _ in scored_pairs:
        predicted_hit = 1 if similarity_score >= threshold else 0

        if predicted_hit == 1 and is_paraphrase == 1:
            true_positives += 1
        elif predicted_hit == 1 and is_paraphrase == 0:
            false_positives += 1
        elif predicted_hit == 0 and is_paraphrase == 0:
            true_negatives += 1
        else:
            false_negatives += 1


    # Out of everything the cache predicted as a “hit,” how many were paraphrases.
    precision_score = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )

    # Out of all true paraphrases, how many the cache successfully marked as a "hit.”
    recall_rate = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )

    return (
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        precision_score,
        recall_rate,
    )


print(
    "threshold  true_positives  false_positives  true_negatives  false_negatives  "
    "precision_score  recall_rate"
)
best_result = None

for threshold_step in range(30, 91):  # 0.30 to 0.90
    threshold = threshold_step / 100
    (
        true_positives,
        false_positives,
        true_negatives,
        false_negatives,
        precision_score,
        recall_rate,
    ) = compute_metrics_at_threshold(threshold)

    print(
        f"{threshold:>8.2f}  "
        f"{true_positives:>14d}  {false_positives:>15d}  {true_negatives:>14d}  "
        f"{false_negatives:>15d}  {precision_score:>15.3f}  {recall_rate:>11.3f}"
    )

    if best_result is None or (
        recall_rate > best_result["recall_rate"]
        or (
            recall_rate == best_result["recall_rate"]
            and precision_score > best_result["precision_score"]
        )
    ):
        best_result = {
            "threshold": threshold,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "precision_score": precision_score,
            "recall_rate": recall_rate,
        }

print("\nBest threshold by recall_rate (tie-break: precision_score):")
print(
    f"threshold={best_result['threshold']:.2f}, "
    f"precision_score={best_result['precision_score']:.3f}, "
    f"recall_rate={best_result['recall_rate']:.3f}, "
    f"true_positives={best_result['true_positives']}, "
    f"false_positives={best_result['false_positives']}, "
    f"true_negatives={best_result['true_negatives']}, "
    f"false_negatives={best_result['false_negatives']}"
)
