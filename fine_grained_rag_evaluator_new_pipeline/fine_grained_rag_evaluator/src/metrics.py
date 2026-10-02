import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.stats import spearmanr, pearsonr, kendalltau

def retrieval_recall(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return None
    top = set(retrieved_ids[:k])
    rel = set(relevant_ids)
    return len(top & rel) / len(rel)

def reciprocal_rank(retrieved_ids, relevant_ids):
    rel = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0

def faithfulness_from_claims(verifications):
    if not verifications:
        return None
    supported = sum(x["label"] == "entailment" for x in verifications)
    return supported / len(verifications)

def hallucination_rate(verifications):
    if not verifications:
        return None
    unsupported = sum(
        x["label"] in {"contradiction", "neutral"} for x in verifications
    )
    return unsupported / len(verifications)

def completeness(answer_claims, reference_claims, nli_model, threshold=0.5):
    if not reference_claims:
        return None
    covered = 0
    for ref in reference_claims:
        best = 0.0
        for claim in answer_claims:
            pred = nli_model.predict(claim, ref)
            best = max(best, pred["scores"].get("entailment", 0.0))
        if best >= threshold:
            covered += 1
    return covered / len(reference_claims)

def agreement_report(y_true, y_pred):
    result = {}
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return result
    result["accuracy"] = float(accuracy_score(y_true, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    result["macro_precision"] = float(p)
    result["macro_recall"] = float(r)
    result["macro_f1"] = float(f1)
    return result

def correlation_report(human, automated):
    if len(human) < 2 or len(human) != len(automated):
        return {}
    return {
        "spearman": float(spearmanr(human, automated).statistic),
        "pearson": float(pearsonr(human, automated).statistic),
        "kendall": float(kendalltau(human, automated).statistic)
    }
