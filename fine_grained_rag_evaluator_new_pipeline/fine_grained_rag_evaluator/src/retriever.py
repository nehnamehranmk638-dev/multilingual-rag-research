import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def tokenize(text):
    return re.findall(r"\w+", text.lower())

class HybridRetriever:
    def __init__(self, corpus, dense_model_name="intfloat/multilingual-e5-base", rrf_k=60):
        self.corpus = corpus
        self.texts = [x["text"] for x in corpus]
        self.ids = [x["id"] for x in corpus]
        self.bm25 = BM25Okapi([tokenize(x) for x in self.texts])
        self.model = SentenceTransformer(dense_model_name)
        self.doc_embeddings = self.model.encode(
            ["passage: " + x for x in self.texts],
            normalize_embeddings=True,
            show_progress_bar=True
        )
        self.rrf_k = rrf_k

    def bm25_search(self, query, top_k=5):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def dense_search(self, query, top_k=5):
        q = self.model.encode(
            ["query: " + query],
            normalize_embeddings=True
        )[0]
        scores = self.doc_embeddings @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def hybrid_search(self, query, top_k=5):
        bm = self.bm25_search(query, top_k=max(top_k, 10))
        de = self.dense_search(query, top_k=max(top_k, 10))

        fused = {}
        for rank, (idx, _) in enumerate(bm, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (idx, _) in enumerate(de, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        order = sorted(fused, key=fused.get, reverse=True)[:top_k]
        return [
            {
                "rank": rank,
                "id": self.ids[i],
                "text": self.texts[i],
                "rrf_score": float(fused[i])
            }
            for rank, i in enumerate(order, start=1)
        ]
