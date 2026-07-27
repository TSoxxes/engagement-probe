"""Small dependency-free text baselines used by the Round 2 audit."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

import numpy as np


TOKEN_RE = re.compile(r"[a-z][a-z0-9']*", re.IGNORECASE)


def ngrams(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return tokens + [
        f"{left}__{right}" for left, right in zip(tokens, tokens[1:])
    ]


def tfidf_features(
    train_texts: Sequence[str],
    test_texts: Sequence[str],
    min_document_frequency: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit word unigram/bigram TF-IDF on train and transform both sets."""
    train_documents = [set(ngrams(text)) for text in train_texts]
    document_frequency = Counter(
        token for document in train_documents for token in document
    )
    vocabulary = {
        token: index
        for index, token in enumerate(
            sorted(
                token
                for token, count in document_frequency.items()
                if count >= min_document_frequency
            )
        )
    }
    if not vocabulary:
        return (
            np.zeros((len(train_texts), 1), dtype=float),
            np.zeros((len(test_texts), 1), dtype=float),
        )
    inverse_document_frequency = np.ones(len(vocabulary), dtype=float)
    for token, index in vocabulary.items():
        inverse_document_frequency[index] = np.log(
            (1 + len(train_texts)) / (1 + document_frequency[token])
        ) + 1

    def transform(texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(vocabulary)), dtype=float)
        for row_index, text in enumerate(texts):
            counts = Counter(ngrams(text))
            for token, count in counts.items():
                column_index = vocabulary.get(token)
                if column_index is not None:
                    matrix[row_index, column_index] = (
                        1 + np.log(count)
                    ) * inverse_document_frequency[column_index]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1, norms)

    return transform(train_texts), transform(test_texts)


def ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """Fit ridge with an intercept and return predictions."""
    train_design = np.column_stack(
        [np.ones(len(train_features)), train_features]
    )
    test_design = np.column_stack(
        [np.ones(len(test_features)), test_features]
    )
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0
    coefficients = np.linalg.pinv(
        train_design.T @ train_design + penalty
    ) @ train_design.T @ train_targets
    return test_design @ coefficients


def lolo_text_regression(
    texts: Sequence[str],
    targets: np.ndarray,
    groups: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    predictions = np.full(len(texts), np.nan, dtype=float)
    text_array = np.asarray(texts, dtype=object)
    for held_out in np.unique(groups):
        train = groups != held_out
        test = ~train
        train_features, test_features = tfidf_features(
            text_array[train].tolist(),
            text_array[test].tolist(),
        )
        predictions[test] = ridge_predict(
            train_features,
            targets[train],
            test_features,
            alpha,
        )
    return predictions


def lolo_text_classification(
    texts: Sequence[str],
    labels: Sequence[str],
    groups: Sequence[str],
    alpha: float = 1.0,
) -> tuple[np.ndarray, float]:
    """One-vs-rest ridge classification with leave-one-group-out folds."""
    labels_array = np.asarray(labels, dtype=str)
    groups_array = np.asarray(groups, dtype=str)
    classes = np.unique(labels_array)
    predictions = np.empty(len(labels_array), dtype=object)
    text_array = np.asarray(texts, dtype=object)
    for held_out in np.unique(groups_array):
        train = groups_array != held_out
        test = ~train
        train_features, test_features = tfidf_features(
            text_array[train].tolist(),
            text_array[test].tolist(),
        )
        one_hot = np.column_stack(
            [(labels_array[train] == value).astype(float) for value in classes]
        )
        scores = ridge_predict(
            train_features,
            one_hot,
            test_features,
            alpha,
        )
        predictions[test] = classes[np.argmax(scores, axis=1)]
    return predictions.astype(str), float(np.mean(predictions == labels_array))


def surface_features(texts: Sequence[str]) -> np.ndarray:
    """Non-semantic style features for mood/pronoun shortcut checks."""
    first_words = ("what", "why", "how", "which", "is", "will", "can",
                   "please", "give", "explain", "tell", "open", "log",
                   "contact", "access", "submit", "write", "book")
    rows = []
    for text in texts:
        lowered = text.lower().strip()
        tokens = TOKEN_RE.findall(lowered)
        first = tokens[0] if tokens else ""
        rows.append(
            [
                float(lowered.endswith("?")),
                float(" i " in f" {lowered} "),
                float(" me " in f" {lowered} "),
                float(" my " in f" {lowered} "),
                float("for me" in lowered),
                float("on my behalf" in lowered),
                float("please" in lowered),
                min(len(tokens), 40) / 40,
                *[float(first == word) for word in first_words],
            ]
        )
    return np.asarray(rows, dtype=float)
