import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import read_jsonl, build_corpus
from src.retriever import HybridRetriever
from src.generator import QwenGenerator
from src.claims import ClaimExtractor
from src.nli import ClaimEvidenceNLI
from src.evaluator import FineGrainedEvaluator
from src.experiment import run_experiment, save_results

def main():
    cfg = load_config()

    corpus_rows = read_jsonl("data/sample_corpus.jsonl")
    question_rows = read_jsonl("data/sample_questions.jsonl")
    corpus = build_corpus(corpus_rows)

    retriever = HybridRetriever(
        corpus,
        dense_model_name=cfg["retrieval"]["dense_model"],
        rrf_k=cfg["retrieval"]["rrf_k"]
    )

    generator = QwenGenerator(
        cfg["generation"]["model"],
        max_new_tokens=cfg["generation"]["max_new_tokens"],
        temperature=cfg["generation"]["temperature"],
        do_sample=cfg["generation"]["do_sample"]
    )

    claim_extractor = ClaimExtractor(
        generator,
        max_claims=cfg["evaluation"]["max_claims"]
    )

    nli = ClaimEvidenceNLI(cfg["nli"]["model"])

    evaluator = FineGrainedEvaluator(
        claim_extractor,
        nli,
        completeness_threshold=cfg["evaluation"]["completeness_threshold"]
    )

    results = run_experiment(
        question_rows,
        retriever,
        generator,
        evaluator,
        top_k=cfg["retrieval"]["top_k"]
    )

    save_results(results, "results.json")

    for r in results:
        print("\nQUESTION:", r["question"])
        print("ANSWER:", r["answer"])
        print("CLAIMS:", r["claims"])
        print("FAITHFULNESS:", r["faithfulness"])
        print("HALLUCINATION RATE:", r["hallucination_rate"])
        print("COMPLETENESS:", r["completeness"])

if __name__ == "__main__":
    main()
