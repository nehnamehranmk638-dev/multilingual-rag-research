# Malayalam Phase 1 — Experiment Summary

## 1. Retrieval Comparison

We evaluated BM25, multilingual dense retrieval (`multilingual-e5-base`), and Hybrid RRF
on the 100-question Malayalam evaluation set (IndicWiki corpus, 247 passages).

| Metric     | BM25   | Dense (E5) | Hybrid (RRF) |
|------------|--------|------------|---------------|
| Recall@1   | 0.65   | 0.66       | 0.68          |
| Recall@3   | 0.75   | 0.84       | 0.87          |
| Recall@5   | 0.82   | 0.90       | 0.91          |
| Recall@10  | 0.89  | 0.93      | 0.97         |
| MRR        | 0.7152 | 0.7567   | 0.7832      |

## 2. Generation and Gold-Passage Ceiling

Model: `Qwen/Qwen2.5-7B-Instruct`. Malayalam-language instruction prompt with script-aware normalizer.

| Condition               | EM     | F1     | Contains-match |
|-------------------------|--------|--------|----------------|
| Hybrid retrieval        | 0.0600 | 0.1312 | 0.1000       |
| Gold passage (ceiling)  | 0.0600 | 0.1365 | 0.1100       |

Refusals: 56 / 100

## 3. Manual Faithfulness Evaluation (30 sampled)

| Label         | Count | % |
|---------------|-------|---|
| Supported     | 0    | 0.0% |
| Not supported | 30    | 100.0% |
| Partial       | 0    | 0.0% |
| Refused       | 0    | 0.0% |

Error types: {'generator_ignored_context': 13, 'retrieval_missed': 17}

## 4. Comparison with English Baseline

| Metric          | English | Malayalam | Drop / Difference  |
|-----------------|---------|-----------|--------------------|
| BM25 Recall@5   | 0.91    | 0.82      | 9 pp  |
| Dense Recall@5  | 0.95    | 0.90      | 5 pp  |
| Hybrid Recall@5 | 0.95    | 0.91      | 4 pp  |
| Hybrid MRR      | 0.892   | 0.783      | 10.9 pp  |
| Generation EM   | 0.61    | 0.06      | 55 pp  |
| Generation F1   | 0.71    | 0.13      | 58 pp  |
| Faithfulness %  | 56.7%   | 0.0%      | -     |

