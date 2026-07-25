"""Small generation helpers that can be tested without loading a model."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def normalize_token_ids(token_ids: Any) -> list[int]:
    """Return an EOS configuration as a plain list of integer token IDs."""
    if token_ids is None:
        return []
    if isinstance(token_ids, int):
        return [token_ids]
    if isinstance(token_ids, Iterable) and not isinstance(token_ids, (str, bytes)):
        return [int(token_id) for token_id in token_ids]
    return [int(token_ids)]


def resolve_eos_token_id(model: Any, tokenizer: Any) -> int | list[int] | None:
    """Prefer the model's full generation stop list over tokenizer EOS alone."""
    candidates = (
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


def infer_finish_reason(
    token_count: int,
    max_new_tokens: int,
    final_token_id: int | None,
    eos_token_id: int | list[int] | None,
) -> str:
    """Classify the only three stopping outcomes used by this prototype."""
    if final_token_id is not None and final_token_id in normalize_token_ids(eos_token_id):
        return "eos"
    if token_count >= max_new_tokens:
        return "length"
    return "other"
