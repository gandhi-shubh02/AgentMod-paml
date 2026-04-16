import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "this",
    "these",
    "those",
    "you",
    "your",
    "yours",
    "they",
    "them",
    "their",
    "we",
    "our",
    "ours",
    "i",
    "me",
    "my",
    "mine",
    "or",
    "if",
    "then",
    "than",
    "so",
    "but",
    "not",
    "no",
    "do",
    "does",
    "did",
    "have",
    "had",
    "having",
}


URL_RE = re.compile(r"http\S+")
HTML_RE = re.compile(r"<.*?>")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = ""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text).strip()
    tokens = [tok for tok in text.split(" ") if tok and tok not in STOPWORDS]
    return " ".join(tokens)


def apply_text_preprocessing(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).apply(clean_text)


def build_binary_toxic_label(df: pd.DataFrame) -> np.ndarray:
    label_sum = df[LABEL_COLUMNS].sum(axis=1).to_numpy(dtype=np.int64)
    return (label_sum > 0).astype(np.int64)


def get_borderline_indices(df: pd.DataFrame) -> np.ndarray:
    label_sum = df[LABEL_COLUMNS].sum(axis=1).to_numpy(dtype=np.int64)
    return np.where(label_sum == 1)[0]


def simulate_user_ids(n_rows: int, n_users: int = 500) -> np.ndarray:
    return np.arange(n_rows, dtype=np.int64) % n_users


def compute_user_features(
    cleaned_texts: List[str], y: np.ndarray, user_ids: np.ndarray
) -> Dict[int, np.ndarray]:
    token_lengths = np.array([len(t.split()) for t in cleaned_texts], dtype=np.float64)
    user_features: Dict[int, np.ndarray] = {}
    unique_users = np.unique(user_ids)
    for uid in unique_users:
        idx = np.where(user_ids == uid)[0]
        count = float(idx.size)
        toxic_rate = float(y[idx].mean()) if idx.size > 0 else 0.0
        mean_len = float(token_lengths[idx].mean()) if idx.size > 0 else 0.0
        std_len = float(token_lengths[idx].std()) if idx.size > 0 else 0.0
        user_features[int(uid)] = np.array([count, toxic_rate, mean_len, std_len], dtype=np.float64)
    return user_features


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stratified_split_indices(
    y: np.ndarray,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitData:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []

    for label in [0, 1]:
        idx = np.where(y == label)[0].copy()
        rng.shuffle(idx)
        n = idx.size
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_test = n - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n - n_train

        train_parts.append(idx[:n_train])
        val_parts.append(idx[n_train : n_train + n_val])
        test_parts.append(idx[n_train + n_val : n_train + n_val + n_test])

    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    test_idx = np.concatenate(test_parts)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return SplitData(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def oversample_toxic_to_ratio(
    X: np.ndarray, y: np.ndarray, target_toxic_to_non_toxic_ratio: float = 1.0 / 3.0, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    toxic_idx = np.where(y == 1)[0]
    non_toxic_idx = np.where(y == 0)[0]

    target_toxic_count = int(np.ceil(non_toxic_idx.size * target_toxic_to_non_toxic_ratio))
    if toxic_idx.size == 0 or toxic_idx.size >= target_toxic_count:
        return X, y

    n_extra = target_toxic_count - toxic_idx.size
    extra_idx = rng.choice(toxic_idx, size=n_extra, replace=True)
    selected_idx = np.concatenate([np.arange(y.size), extra_idx])
    rng.shuffle(selected_idx)
    return X[selected_idx], y[selected_idx]


class MinMaxScaler:
    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        X = np.asarray(X, dtype=np.float64)
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        denom = self.max_ - self.min_
        denom[denom == 0.0] = 1.0
        self.scale_ = denom

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transform.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.min_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def save(self, path: str) -> None:
        if self.min_ is None or self.max_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before save.")
        np.savez(path, min_=self.min_, max_=self.max_, scale_=self.scale_)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.min_ = data["min_"]
        self.max_ = data["max_"]
        self.scale_ = data["scale_"]


def build_user_feature_matrix(user_features: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    user_ids = np.array(sorted(user_features.keys()), dtype=np.int64)
    matrix = np.vstack([user_features[int(uid)] for uid in user_ids]).astype(np.float64)
    return user_ids, matrix

