import json
from tqdm import tqdm
from .data import get_reference_claims
from .metrics import retrieval_recall, reciprocal_rank

def run_experiment(rows, retriever, generator, evaluator, top_k=5):
    outputs = []

    for row in tqdm(rows):
        question = row["question"]
        evidence = retriever.hybrid_search(question, top_k=top_k)
        answer = generator.generate(question, evidence)

        result = evaluator.evaluate(
            answer,
            evidence,
            reference_claims=get_reference_claims(row)
        )

        relevant_ids = row.get("relevant_ids", [])
        retrieved_ids = [x["id"] for x in evidence]

        result["question_id"] = row.get("id")
        result["question"] = question
        result["retrieved_ids"] = retrieved_ids
        result["recall_at_k"] = retrieval_recall(
            retrieved_ids, relevant_ids, top_k
        ) if relevant_ids else None
        result["rr"] = reciprocal_rank(
            retrieved_ids, relevant_ids
        ) if relevant_ids else None

        outputs.append(result)

    return outputs

def save_results(results, path="results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
