
# Step 8 — Experiment Summary

## 1. Retrieval Comparison

We evaluated three retrieval approaches on the 100-question evaluation set:

- BM25
- Dense retrieval using `intfloat/multilingual-e5-base`
- Hybrid retrieval using Reciprocal Rank Fusion (RRF)

### Recall@5

- BM25: 0.91
- Dense (E5): 0.95
- Hybrid (RRF): 0.95

### MRR

- BM25: 0.845968
- Dense (E5): 0.846111
- Hybrid (RRF): 0.891940

The hybrid method achieved a Recall@5 of 0.95 and an MRR of 0.891940 in the evaluation.

The complete Recall@k results were:

| Metric | BM25 | Dense (E5) | Hybrid (RRF) |
|---|---:|---:|---:|
| Recall@1 | 0.79 | 0.76 | 0.84 |
| Recall@3 | 0.88 | 0.95 | 0.93 |
| Recall@5 | 0.91 | 0.95 | 0.95 |
| Recall@10 | 0.95 | 0.97 | 0.97 |

## 2. Generation and Gold-Passage Ceiling

Using hybrid retrieval, the generation results were:

- EM: 0.61
- F1: 0.712394
- Contains-match: 0.76
- Refusals: 0

The model used for generation was `Qwen/Qwen2.5-1.5B-Instruct`.

When the gold passage was supplied directly:

- EM: 0.71
- F1: 0.811669

The difference between hybrid retrieval and gold-passage conditions was:

- EM gap: 10.0 percentage points
- F1 gap: 9.93 percentage points

Providing the gold passage increased EM from 0.61 to 0.71 and F1 from 0.712394 to 0.811669. This indicates that retrieval quality contributes substantially to the observed generation performance gap.

## 3. Manual Faithfulness Evaluation

We manually evaluated 30 sampled cases.

- Supported: 56.7%
- Partial: 3.3%
- Not supported: 40.0%
- Refused: 0%

The observed error types were:

- `retrieval_missed`: 10 cases
- `generator_ignored_context`: 2 cases

No cases were labeled as `made_up_detail` or `correct_but_different_wording`.

Among the unsupported cases in this sample, retrieval misses were the dominant failure mode.

## 4. Implications for Multilingual Evaluation

These experiments were conducted using the English evaluation setup and provide a baseline for the later multilingual phase.

The results motivate evaluating retrieval and faithfulness for Malayalam, Tamil, Telugu, and other target languages rather than assuming that the English results will transfer directly.

The multilingual experiments can help determine whether the retrieval and faithfulness issues observed in the English baseline also occur across the target languages.

## 5. Limitations

The manual faithfulness evaluation used a sample of 30 questions and was performed by a single annotator. Inter-annotator agreement was therefore not calculated.

The faithfulness percentages should be interpreted as findings from this sample rather than estimates of the entire evaluation set.

The gold-passage experiment measures the difference between hybrid retrieval and providing the correct passage. This difference should not be interpreted as a precise causal decomposition of all system errors.
