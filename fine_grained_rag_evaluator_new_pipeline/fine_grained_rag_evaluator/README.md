# Fine-Grained RAG Evaluator

A research-oriented starter implementation for the proposed pipeline:

Query
→ Hybrid Retrieval (BM25 + multilingual E5 + RRF)
→ Top-K Evidence
→ Qwen2.5-Instruct
→ Claim Extraction
→ Atomic Claims
→ Claim–Evidence NLI
→ Entailment / Contradiction / Neutral
→ Faithfulness / Hallucination / Completeness

## Research phases

### Phase 1 — English baseline
Default configuration:
- Dataset format: JSONL
- Retriever: BM25 + intfloat/multilingual-e5-base + Reciprocal Rank Fusion
- Generator: Qwen/Qwen2.5-1.5B-Instruct
- Claim extraction: same Qwen instruct model
- NLI: cross-encoder/nli-deberta-v3-base
- Metrics: Recall@K, MRR, claim-level faithfulness, hallucination, completeness

### Phase 2 — Indian-language evaluation
Change the configuration to an Indic/multilingual dataset and an NLI model fine-tuned for the target language. Do NOT assume that a pretrained encoder such as MuRIL is automatically an NLI classifier.

The repository is designed so that English, Hindi, Malayalam and Tamil experiments can use the same evaluator interface.

## Important research point

The retrieval component is a supporting experiment. The main research question is evaluator reliability:

1. Generate the same RAG outputs.
2. Evaluate them with existing evaluators where available.
3. Compare automated scores against human judgments.
4. Repeat across English and Indian languages.
5. Perform ablations and error analysis.

Do not claim improved results until they are measured.

## Folder structure

```text
fine_grained_rag_evaluator/
├── README.md
├── requirements.txt
├── configs/
│   └── config.yaml
├── data/
│   ├── sample_corpus.jsonl
│   ├── sample_questions.jsonl
│   └── human_annotations_template.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── retriever.py
│   ├── generator.py
│   ├── claims.py
│   ├── nli.py
│   ├── metrics.py
│   ├── evaluator.py
│   └── experiment.py
├── scripts/
│   └── run_pipeline.py
└── notebooks/
    └── research_pipeline.ipynb
```

## Install

```bash
pip install -r requirements.txt
```

For GPU/Colab, use a CUDA-compatible PyTorch installation first if needed.

## Run the sample pipeline

```bash
python scripts/run_pipeline.py
```

The first run downloads Hugging Face models and can take time.

## Use your own data

### corpus.jsonl

Each line:

```json
{"id":"doc1","text":"Your evidence passage here."}
```

### questions.jsonl

Each line:

```json
{"id":"q1","question":"Your question","reference_answer":"Optional reference answer","reference_claims":["Optional atomic reference claim"]}
```

`reference_claims` are needed for the completeness metric. If they are unavailable, completeness is reported as null rather than invented.

## Research experiments

### Retrieval
Compare:
- BM25
- Dense E5
- Hybrid RRF

Metrics:
- Recall@1
- Recall@5
- Recall@10
- MRR

### Evaluator
Compare automated evaluation against human annotations.

Recommended human annotation fields:
- claim
- evidence
- label: entailment / contradiction / neutral
- answer-level faithfulness
- answer-level hallucination
- completeness

### Cross-language
Run the same pipeline for:
- English
- Hindi
- Malayalam
- Tamil

Record:
- retrieval metrics
- claim extraction failures
- NLI failures
- human/evaluator agreement
- hallucination detection F1
- completeness agreement

## Warning

This is a research starter repository, not a finished paper result. Model choices, thresholds, prompts, annotation protocol, statistical tests and benchmark splits must be validated before reporting results.
