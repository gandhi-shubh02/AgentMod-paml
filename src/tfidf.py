from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


class TFIDFVectorizer:
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: np.ndarray | None = None
        self.feature_names_: List[str] = []

    def _tokenize(self, doc: str) -> List[str]:
        if not isinstance(doc, str):
            return []
        return [tok for tok in doc.split() if tok]

    def _generate_ngrams(self, tokens: List[str]) -> List[str]:
        n_min, n_max = self.ngram_range
        terms: List[str] = []
        for n in range(n_min, n_max + 1):
            if n == 1:
                terms.extend(tokens)
            else:
                if len(tokens) >= n:
                    terms.extend([" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])
        return terms

    def fit(self, corpus: List[str]) -> None:
        doc_freq = Counter()
        n_docs = len(corpus)

        for doc in corpus:
            tokens = self._tokenize(doc)
            terms = set(self._generate_ngrams(tokens))
            doc_freq.update(terms)

        top_terms = [t for t, _ in doc_freq.most_common(self.max_features)]
        self.feature_names_ = top_terms
        self.vocabulary_ = {term: i for i, term in enumerate(top_terms)}

        df = np.array([doc_freq[term] for term in top_terms], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

    def transform(self, corpus: List[str]) -> np.ndarray:
        if self.idf_ is None or not self.vocabulary_:
            raise ValueError("Vectorizer must be fitted before transform.")

        n_docs = len(corpus)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.float64)

        for i, doc in enumerate(corpus):
            tokens = self._tokenize(doc)
            terms = self._generate_ngrams(tokens)
            if not terms:
                continue
            counts = Counter(terms)
            denom = float(len(terms))
            for term, count in counts.items():
                j = self.vocabulary_.get(term)
                if j is not None:
                    tf = count / denom
                    X[i, j] = tf * self.idf_[j]

        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        row_norms[row_norms == 0.0] = 1.0
        X = X / row_norms
        return X

    def fit_transform(self, corpus: List[str]) -> np.ndarray:
        self.fit(corpus)
        return self.transform(corpus)

    def save(self, path: str) -> None:
        vocab_terms = np.array(self.feature_names_, dtype=object)
        np.savez(path, vocab_terms=vocab_terms, idf=self.idf_, max_features=self.max_features, ngram_range=np.array(self.ngram_range))

    def load(self, path: str) -> None:
        data = np.load(path, allow_pickle=True)
        self.feature_names_ = [str(x) for x in data["vocab_terms"].tolist()]
        self.vocabulary_ = {term: i for i, term in enumerate(self.feature_names_)}
        self.idf_ = data["idf"].astype(np.float64)
        self.max_features = int(data["max_features"])
        ng = data["ngram_range"]
        self.ngram_range = (int(ng[0]), int(ng[1]))

