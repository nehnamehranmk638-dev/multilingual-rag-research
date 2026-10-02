from .metrics import (
    faithfulness_from_claims,
    hallucination_rate,
    completeness
)

class FineGrainedEvaluator:
    def __init__(self, claim_extractor, nli_model, completeness_threshold=0.5):
        self.claim_extractor = claim_extractor
        self.nli_model = nli_model
        self.completeness_threshold = completeness_threshold

    def evaluate(self, answer, evidence, reference_claims=None):
        claims = self.claim_extractor.extract(answer)
        verifications = [
            self.nli_model.verify_claim(claim, evidence)
            for claim in claims
        ]
        faithfulness = faithfulness_from_claims(verifications)
        hallucination = hallucination_rate(verifications)

        if reference_claims:
            completeness_score = completeness(
                claims,
                reference_claims,
                self.nli_model,
                self.completeness_threshold
            )
        else:
            completeness_score = None

        return {
            "answer": answer,
            "claims": claims,
            "claim_verifications": verifications,
            "faithfulness": faithfulness,
            "hallucination_rate": hallucination,
            "completeness": completeness_score
        }
