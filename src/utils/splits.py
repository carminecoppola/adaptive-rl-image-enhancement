"""
Deterministic dataset split helpers.
"""

from __future__ import annotations

import random
from collections.abc import Iterable


def build_train_eval_indices(
    dataset_size: int,
    eval_pool_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Build deterministic train/eval index sets from [0, dataset_size).
    """
    if dataset_size <= 0:
        raise ValueError("dataset_size must be > 0")

    eval_pool_size = max(1, min(eval_pool_size, dataset_size))

    all_indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(all_indices)

    eval_indices = sorted(all_indices[:eval_pool_size])
    train_indices = sorted(all_indices[eval_pool_size:])

    if not train_indices:
        train_indices = eval_indices.copy()

    return train_indices, eval_indices


def sample_indices(indices: Iterable[int], k: int, seed: int) -> list[int]:
    """
    Deterministically sample up to k elements from indices.
    """
    pool = list(indices)
    if not pool:
        return []

    k = min(k, len(pool))
    rng = random.Random(seed)
    return rng.sample(pool, k)


def apply_subset_limits(
    train_indices: list[int],
    eval_indices: list[int],
    train_subset_size: int | None,
    eval_subset_size: int | None,
    seed: int,
) -> tuple[list[int], list[int]]:
    """
    Deterministically reduce train/eval pools when subset sizes are provided.

    If a subset size is None or <= 0, keep the full corresponding split.
    """
    train_out = train_indices
    eval_out = eval_indices

    if isinstance(train_subset_size, int) and train_subset_size > 0:
        train_out = sample_indices(train_indices, k=train_subset_size, seed=seed + 11)
        train_out = sorted(train_out)

    if isinstance(eval_subset_size, int) and eval_subset_size > 0:
        eval_out = sample_indices(eval_indices, k=eval_subset_size, seed=seed + 17)
        eval_out = sorted(eval_out)

    if not train_out:
        train_out = eval_out.copy()

    return train_out, eval_out
