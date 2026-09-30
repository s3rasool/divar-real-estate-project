"""Train / Validation / Test split utilities for the Divar modeling workflow.

Task 5 goals implemented here:
- hold out Test FIRST
- split the remaining data into Train and Validation
- keep Test untouched for final evaluation
- use a fixed random_state for reproducibility
- keep split ratios explicit and stable
- compare target distributions across splits
- validate that the splits are disjoint and complete

This module intentionally does NOT choose the target or final feature columns.
Those decisions belong to Tasks 1-4. It consumes X and y after those steps.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    """Container for the three modeling splits."""

    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


def _validate_split_sizes(test_size: float, validation_size: float) -> None:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1:
        raise ValueError(
            "test_size + validation_size must be less than 1 "
            "so that training data remains."
        )


def _prepare_xy(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate alignment and reset to a clean positional index."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows.")

    if not X.index.equals(y.index):
        raise ValueError(
            "X and y indexes are not aligned. Align them before splitting."
        )

    if y.isna().any():
        raise ValueError(
            "y contains missing target values. "
            "Do not impute the target; resolve/filter them before Task 5."
        )

    X_clean = X.reset_index(drop=True).copy()
    y_clean = y.reset_index(drop=True).copy()
    return X_clean, y_clean


def split_train_validation_test(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.15,
    validation_size: float = 0.15,
    random_state: int = 42,
) -> DataSplits:
    """Split data into Train / Validation / Test without touching Test later.

    ``test_size`` and ``validation_size`` are fractions of the FULL dataset.

    The split is deliberately performed in two stages:
    1) Test is held out first.
    2) The remaining data is split into Train and Validation.

    With the defaults the final proportions are 70% / 15% / 15%.

    Notes
    -----
    This is a random split, matching the Task 5 requirement to keep a fixed
    ``random_state``. If Tasks 1-4 later establish that the production problem
    requires a temporal split, this function should be replaced by that
    project-specific split policy rather than mixing the two approaches.
    """
    _validate_split_sizes(test_size, validation_size)
    X, y = _prepare_xy(X, y)

    # Stage 1: isolate Test first.
    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # validation_size is defined as a fraction of the ORIGINAL dataset.
    # Convert it to a fraction of the remaining data.
    validation_relative = validation_size / (1.0 - test_size)

    # Stage 2: split only the remaining data into Train and Validation.
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_remaining,
        y_remaining,
        test_size=validation_relative,
        random_state=random_state,
        shuffle=True,
    )

    splits = DataSplits(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )

    validate_splits(splits, original_size=len(X))
    return splits


def validate_splits(splits: DataSplits, *, original_size: int) -> None:
    """Check size consistency and X/y alignment for every split."""
    split_pairs = {
        "train": (splits.X_train, splits.y_train),
        "validation": (splits.X_validation, splits.y_validation),
        "test": (splits.X_test, splits.y_test),
    }

    for name, (X_part, y_part) in split_pairs.items():
        if len(X_part) != len(y_part):
            raise AssertionError(f"X_{name} and y_{name} lengths do not match.")
        if not X_part.index.equals(y_part.index):
            raise AssertionError(f"X_{name} and y_{name} indexes are not aligned.")

    total = sum(len(X_part) for X_part, _ in split_pairs.values())
    if total != original_size:
        raise AssertionError(
            f"Split sizes sum to {total}, expected {original_size}."
        )

    train_idx = set(splits.X_train.index)
    validation_idx = set(splits.X_validation.index)
    test_idx = set(splits.X_test.index)

    if train_idx & validation_idx:
        raise AssertionError("Train and Validation overlap.")
    if train_idx & test_idx:
        raise AssertionError("Train and Test overlap.")
    if validation_idx & test_idx:
        raise AssertionError("Validation and Test overlap.")


def split_size_summary(splits: DataSplits) -> pd.DataFrame:
    """Return row counts and percentages for the three splits."""
    counts = {
        "train": len(splits.X_train),
        "validation": len(splits.X_validation),
        "test": len(splits.X_test),
    }
    total = sum(counts.values())

    return pd.DataFrame(
        {
            "rows": counts,
            "percentage": {k: 100 * v / total for k, v in counts.items()},
        }
    )


def _target_stats(y: pd.Series) -> dict[str, float]:
    """Compact regression-target diagnostics for one split."""
    return {
        "count": float(y.count()),
        "mean": float(y.mean()),
        "std": float(y.std()),
        "median": float(y.median()),
        "min": float(y.min()),
        "p05": float(y.quantile(0.05)),
        "p95": float(y.quantile(0.95)),
        "max": float(y.max()),
    }


def compare_target_distributions(splits: DataSplits) -> pd.DataFrame:
    """Compare target distributions across Train / Validation / Test.

    This is a sanity check, not a proof that the split is perfect. Large,
    unexplained distribution differences should be investigated before modeling.
    """
    return pd.DataFrame(
        {
            "train": _target_stats(splits.y_train),
            "validation": _target_stats(splits.y_validation),
            "test": _target_stats(splits.y_test),
        }
    ).T
