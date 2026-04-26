from sentence_transformers import SentenceTransformer
import numpy as np

# Format: (left_text, right_text, is_paraphrase)
# is_paraphrase: 1 = should be cache hit, 0 = should be cache miss
LABELED_PAIRS = [
    # Positive pairs (same intent)
    ("What's your return policy?", "How do I return something I bought?", 1),
    ("Can I get a refund?", "How do refunds work?", 1),
    ("Where is my order?", "How can I track my shipment?", 1),
    ("I need to change my password", "How do I reset my password?", 1),
    ("Do you offer free shipping?", "Is shipping free?", 1),
    ("How long does delivery take?", "What's your typical shipping time?", 1),
    ("Can I update my shipping address?", "How do I change my delivery address?", 1),
    ("Can I cancel my order?", "How do I cancel a purchase?", 1),
    ("How can I contact support?", "How do I reach customer service?", 1),
    ("Can I talk to a human agent?", "Can I speak with a support representative?", 1),
    ("Do you have a student discount?", "Is there a discount for students?", 1),
    ("Do you ship internationally?", "Can you ship overseas?", 1),
    ("What payment methods do you accept?", "How can I pay?", 1),
    ("How do I apply a promo code?", "How can I use a discount code?", 1),
    ("What are your business hours?", "When are you open?", 1),
    ("Can I exchange an item?", "How do exchanges work?", 1),
    ("How do I delete my account?", "How can I close my account?", 1),
    ("Where can I download my invoice?", "How do I get a billing receipt?", 1),
    ("My package is late", "Why has my order not arrived yet?", 1),
    ("How do I check my refund status?", "Where can I see my refund progress?", 1),
    ("Hi", "Hello", 1),
    ("Hello", "Hey", 1),
    ("I am testing something.", "I'm just testing this.", 1),
    ("How do I go from point A to point B on feet?", "How can I walk from point A to point B?", 1),
    ("How do I go from point A to point B on fett?", "How do I go from point A to point B on feet?", 1),
    ("I forgot my password", "I cannot sign in and need a password reset", 1),
    ("Can I return this without a receipt?", "Do I need a receipt for returns?", 1),
    ("How can I contact support?", "What is the best way to reach support?", 1),
    ("Can I use PayPal?", "Do you accept PayPal payments?", 1),
    ("How long does shipping take?", "What is the delivery ETA?", 1),

    # Negative pairs (different intent)
    ("What's your return policy?", "How do I reset my password?", 0),
    ("Can I get a refund?", "Do you have a student discount?", 0),
    ("Where is my order?", "How do I delete my account?", 0),
    ("Do you offer free shipping?", "What are your business hours?", 0),
    ("How long does delivery take?", "What payment methods do you accept?", 0),
    ("Can I cancel my order?", "How do I close my account?", 0),
    ("How can I contact support?", "How do I apply a promo code?", 0),
    ("Can I talk to a human agent?", "Do you ship internationally?", 0),
    ("Do you have a student discount?", "How do I reset my password?", 0),
    ("Can you ship overseas?", "What's your return policy?", 0),
    ("How can I pay?", "Where is my order?", 0),
    ("How do I apply a promo code?", "What are your business hours?", 0),
    ("Hi", "How do I reset my password?", 0),
    ("Hello", "Where is my order?", 0),
    ("What's up?", "Can I return this item?", 0),
    ("How can I walk from point A to point B?", "How can I track my shipment?", 0),
    ("I want to return this item", "I do not want to return this item", 0),
    ("Can I get a refund?", "Can I get free shipping?", 0),
    ("How do I delete my account?", "How do I change my shipping address?", 0),
    ("How do exchanges work?", "How do I reset my password?", 0),
    ("How do I check my refund status?", "Can I talk to a human agent?", 0),
    ("How do I walk to work?", "How do I track my shipment?", 0),
    ("Can I pay with PayPal?", "Can I return this without a receipt?", 0),
    ("When are you open?", "How do I change my password?", 0),
    ("Where can I download my invoice?", "Where is my order?", 0),
    ("How can I reach support?", "Can I cancel my order?", 0),
    ("Can I use a promo code?", "Do you ship internationally?", 0),
    ("How long does shipping take?", "How do I apply a promo code?", 0),
    ("Can I return this item?", "How do I close my account?", 0),
    ("Can you ship overseas?", "How can I get a billing receipt?", 0),
]

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUALITY_MIN_PRECISION = 0.90


def score_pairs(labeled_pairs):
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    unique_texts = sorted(
        {text for left_text, right_text, _ in labeled_pairs for text in (left_text, right_text)}
    )
    embeddings = embedding_model.encode(unique_texts, normalize_embeddings=True)
    text_to_index = {text: index for index, text in enumerate(unique_texts)}

    scored_pairs = []
    for left_text, right_text, is_paraphrase in labeled_pairs:
        similarity_score = float(
            np.dot(embeddings[text_to_index[left_text]], embeddings[text_to_index[right_text]])
        )
        scored_pairs.append((left_text, right_text, is_paraphrase, similarity_score))

    return scored_pairs


def compute_metrics_at_threshold(scored_pairs, threshold):
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    for _, _, is_paraphrase, similarity_score in scored_pairs:
        predicted_hit = 1 if similarity_score >= threshold else 0

        if predicted_hit == 1 and is_paraphrase == 1:
            true_positives += 1
        elif predicted_hit == 1 and is_paraphrase == 0:
            false_positives += 1
        elif predicted_hit == 0 and is_paraphrase == 0:
            true_negatives += 1
        else:
            false_negatives += 1

    precision_score = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives)
        else 0.0
    )
    recall_rate = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives)
        else 0.0
    )
    f1_score = (
        (2 * precision_score * recall_rate) / (precision_score + recall_rate)
        if (precision_score + recall_rate)
        else 0.0
    )
    accuracy = (true_positives + true_negatives) / len(scored_pairs)

    return {
        "threshold": threshold,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision_score": precision_score,
        "recall_rate": recall_rate,
        "f1_score": f1_score,
        "accuracy": accuracy,
    }


def choose_best_by_f1(candidate_metrics):
    return max(
        candidate_metrics,
        key=lambda metrics: (
            metrics["f1_score"],
            metrics["precision_score"],
            metrics["recall_rate"],
            metrics["threshold"],
        ),
    )


def choose_precision_first(candidate_metrics):
    precision_safe = [
        metrics
        for metrics in candidate_metrics
        if metrics["precision_score"] >= QUALITY_MIN_PRECISION
    ]
    if not precision_safe:
        return None

    return max(
        precision_safe,
        key=lambda metrics: (
            metrics["recall_rate"],
            metrics["f1_score"],
            metrics["precision_score"],
            metrics["threshold"],
        ),
    )


def show_misclassifications(scored_pairs, threshold, limit=8):
    mistakes = []
    for left_text, right_text, is_paraphrase, similarity_score in scored_pairs:
        predicted_hit = 1 if similarity_score >= threshold else 0
        if predicted_hit != is_paraphrase:
            mistakes.append((similarity_score, is_paraphrase, predicted_hit, left_text, right_text))

    # Show the most ambiguous mistakes first (closest to threshold)
    mistakes.sort(key=lambda item: abs(item[0] - threshold))

    print(f"\nMisclassified pairs at threshold={threshold:.2f} (showing up to {limit}):")
    if not mistakes:
        print("  none")
        return

    for similarity_score, is_paraphrase, predicted_hit, left_text, right_text in mistakes[:limit]:
        expected = "hit" if is_paraphrase == 1 else "miss"
        predicted = "hit" if predicted_hit == 1 else "miss"
        print(
            f"  score={similarity_score:.3f} expected={expected} predicted={predicted}\n"
            f"    Q1: {left_text}\n"
            f"    Q2: {right_text}"
        )


def main():
    scored_pairs = score_pairs(LABELED_PAIRS)
    total_pairs = len(scored_pairs)
    total_positive = sum(1 for _, _, is_paraphrase, _ in scored_pairs if is_paraphrase == 1)
    total_negative = total_pairs - total_positive

    print(f"model: {EMBEDDING_MODEL_NAME}")
    print(
        f"dataset size: {total_pairs} pairs ({total_positive} positive / {total_negative} negative)"
    )
    print(
        "threshold  TP  FP  TN  FN  precision  recall  f1  accuracy"
    )

    candidate_metrics = []
    for threshold_step in range(20, 91):  # 0.20 to 0.90
        threshold = threshold_step / 100
        metrics = compute_metrics_at_threshold(scored_pairs, threshold)
        candidate_metrics.append(metrics)
        print(
            f"{threshold:>8.2f}  "
            f"{metrics['true_positives']:>2d}  {metrics['false_positives']:>2d}  "
            f"{metrics['true_negatives']:>2d}  {metrics['false_negatives']:>2d}  "
            f"{metrics['precision_score']:>9.3f}  {metrics['recall_rate']:>6.3f}  "
            f"{metrics['f1_score']:>4.3f}  {metrics['accuracy']:>8.3f}"
        )

    best_by_f1 = choose_best_by_f1(candidate_metrics)
    precision_first = choose_precision_first(candidate_metrics)

    print("\nBest threshold by f1_score:")
    print(
        f"threshold={best_by_f1['threshold']:.2f}, "
        f"TP={best_by_f1['true_positives']}, FP={best_by_f1['false_positives']}, "
        f"TN={best_by_f1['true_negatives']}, FN={best_by_f1['false_negatives']}, "
        f"precision={best_by_f1['precision_score']:.3f}, "
        f"recall={best_by_f1['recall_rate']:.3f}, "
        f"f1={best_by_f1['f1_score']:.3f}, "
        f"accuracy={best_by_f1['accuracy']:.3f}"
    )

    if precision_first is not None:
        print(
            f"\nPrecision-first recommendation (precision >= {QUALITY_MIN_PRECISION:.2f}):"
        )
        print(
            f"threshold={precision_first['threshold']:.2f}, "
            f"TP={precision_first['true_positives']}, FP={precision_first['false_positives']}, "
            f"TN={precision_first['true_negatives']}, FN={precision_first['false_negatives']}, "
            f"precision={precision_first['precision_score']:.3f}, "
            f"recall={precision_first['recall_rate']:.3f}, "
            f"f1={precision_first['f1_score']:.3f}, "
            f"accuracy={precision_first['accuracy']:.3f}"
        )
    else:
        print(
            f"\nNo threshold reached precision >= {QUALITY_MIN_PRECISION:.2f}. "
            "Use best-by-F1 or lower the precision guardrail."
        )

    show_misclassifications(scored_pairs, best_by_f1["threshold"], limit=8)


if __name__ == "__main__":
    main()
