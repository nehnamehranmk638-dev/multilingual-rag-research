\# Step 7 — Manual Faithfulness Findings



\## 1. Evaluation setup



We manually evaluated 30 randomly selected questions from the

100-question evaluation set.



For each question, we inspected:

\- the question

\- the gold answer

\- the top-5 hybrid-retrieved passages

\- the generated model answer



Each answer was assigned one of four faithfulness labels:

\- supported

\- partial

\- not\_supported

\- refused



\## 2. Faithfulness results



| Label | Count | Percentage |

|---|---:|---:|

| Supported | 17 | 56.7% |

| Partial | 1 | 3.3% |

| Not supported | 12 | 40.0% |

| Refused | 0 | 0% |

| Total | 30 | 100% |



\## 3. Error type breakdown



| Error type | Count |

|---|---:|

| retrieval\_missed | 10 |

| generator\_ignored\_context | 2 |

| made\_up\_detail | 0 |

| correct\_but\_different\_wording | 0 |



\## 4. EM/F1 versus faithfulness



The manual evaluation demonstrates that EM and F1 do not directly

measure whether an answer is supported by the retrieved evidence.



A faithful answer can receive EM = 0 because of wording differences.

Conversely, a model can potentially obtain the correct gold answer

without the retrieved passages actually supporting that answer.



\## 5. Connection to Phase 2



The observed error types motivate an automatic faithfulness /

hallucination detection component.



The verifier should particularly detect:

\- unsupported answers

\- information added beyond the retrieved evidence

\- cases where the retrieved evidence does not entail the answer



These findings will inform the planned NLI-based verifier for

Malayalam, Tamil, and Telugu.

