
# Malayalam Phase 1 — Experiment Summary

## 1. Retrieval Comparison

We evaluated BM25, multilingual dense retrieval (multilingual-e5-base), and Hybrid RRF
on the 100-question Malayalam evaluation set (IndicWiki corpus, 247 passages).

| Metric     | BM25   | Dense (E5) | Hybrid (RRF) |
|------------|--------|------------|---------------|
| Recall@1   | 0.65   | 0.66       | 0.68          |
| Recall@3   | 0.75   | 0.84       | 0.87          |
| Recall@5   | 0.82   | 0.90       | 0.91          |
| Recall@10  | 0.89  | 0.93      | 0.97         |
| MRR        | 0.7152 | 0.7567   | 0.7832      |

## 2. Generation and Gold-Passage Ceiling

Model: Qwen/Qwen2.5-1.5B-Instruct. Malayalam-language instruction prompt.

| Condition         | EM     | F1     | Contains-match |
|-------------------|--------|--------|----------------|
| Hybrid retrieval  | 0.0000 | 0.0000 | 0.0000       |

Refusals: 0 / 100

## 3. Manual Faithfulness Evaluation (30 sampled)

| Label         | Count | % |
|---------------|-------|---|
| Supported     | 0    | 0.0% |
| Not supported | 30    | 100.0% |
| Partial       | 0    | 0.0% |
| Refused       | 0    | 0.0% |

Error types: {'generator_ignored_context': 13, 'retrieval_missed': 17}

## 4. Comparison with English Baseline

| Metric          | English | Malayalam | Drop  |
|-----------------|---------|-----------|-------|
| BM25 Recall@5   | 0.91    | 0.82      | 9pp  |
| Dense Recall@5  | 0.95    | 0.90      | 5pp  |
| Hybrid Recall@5 | 0.95    | 0.91      | 4pp  |
| Hybrid MRR      | 0.892   | 0.783      | 10.9pp  |
| Generation EM   | 0.61    | 0.00      | 61pp  |
| Faithfulness %  | 56.7%   | 0.0%      | -     |

