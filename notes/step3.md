\# Step 3 — BM25 Retrieval



\## Goal



The goal of this step is to implement \*\*BM25 (Best Matching 25)\*\*, a traditional keyword-based information retrieval method.



For each of the 100 questions, BM25 searches the collection of passages and retrieves the \*\*top-k most relevant passages\*\* based on keyword overlap.



\## What is BM25?



BM25 is a ranking algorithm used to find documents/passages that are most relevant to a query.



It considers:



\- How often query words appear in a passage (\*\*Term Frequency\*\*)

\- How rare those words are across all passages (\*\*Inverse Document Frequency\*\*)

\- The length of the passage



Unlike semantic retrieval, BM25 mainly relies on \*\*lexical/keyword matching\*\*.



\## What we do



1\. Load the questions and passages.

2\. Build a BM25 index over the passages.

3\. Use each question as a query.

4\. Retrieve the top-k passages for each question.

5\. Compare the retrieved passages with the known \*\*gold passage\*\*.

6\. Calculate retrieval metrics:

&#x20;  - \*\*Recall@k\*\* — whether the gold passage appears in the top-k results.

&#x20;  - \*\*MRR (Mean Reciprocal Rank)\*\* — how highly the gold passage is ranked.



\## Why this step is important



BM25 provides a \*\*baseline retrieval method\*\*.



Later, we can compare its performance with semantic/dense retrieval methods to determine whether understanding the meaning of a query improves retrieval compared with simple keyword matching.



\## Expected Output



For each question:



```text

Question

&#x20;   ↓

BM25 search

&#x20;   ↓

Top-k retrieved passages

&#x20;   ↓

Compare with gold passage

&#x20;   ↓

Recall@k + MRR

