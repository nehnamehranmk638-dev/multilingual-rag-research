
from sentence_transformers import CrossEncoder

class ClaimEvidenceNLI:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base"):
        self.model = CrossEncoder(model_name)
        self.labels = ["contradiction", "entailment", "neutral"]

    def predict(self, claim, evidence):
        scores = self.model.predict(
            [(evidence, claim)],
            apply_softmax=True
        )[0]
        idx = int(scores.argmax())
        return {
            "label": self.labels[idx],
            "scores": {
                self.labels[i]: float(scores[i])
                for i in range(len(self.labels))
            }
        }

    def verify_claim(self, claim, evidence_list):
        results = []
        for ev in evidence_list:
            p = self.predict(claim, ev["text"])
            results.append({
                "evidence_id": ev["id"],
                "evidence": ev["text"],
                **p
            })

        if not results:
            return {
                "claim": claim,
                "label": "neutral",
                "evidence": None,
                "results": []
            }

        priority = {"entailment": 2, "contradiction": 1, "neutral": 0}
        best = max(
            results,
            key=lambda x: (
                priority[x["label"]],
                x["scores"][x["label"]]
            )
        )

        return {
            "claim": claim,
            "label": best["label"],
            "evidence": best["evidence_id"],
            "results": results
        }
