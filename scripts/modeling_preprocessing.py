"""Modeling preprocessing utilities for the Divar real-estate project.

Task 6 goals implemented here:
- missing-value handling
- numeric preparation
- categorical encoding
- optional scaling for models that need it
- fit only on training data
- sklearn Pipeline + ColumnTransformer
- validation/test are transform-only

This module intentionally does NOT choose the target, final features, or split rows.
Those decisions belong to Tasks 1-5. It consumes their outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class FeatureGroups:
    """Column groups inferred only from X_train dtypes."""

    numeric: list[str]
    categorical: list[str]
    boolean: list[str]

    @property
    def all_columns(self) -> list[str]:
        return self.numeric + self.categorical + self.boolean


def infer_feature_groups(X_train: pd.DataFrame) -> FeatureGroups:
    """Infer numeric/categorical/boolean feature groups from training data only.

    Notes
    -----
    - Boolean columns are handled separately so they are not accidentally scaled.
    - String/object/category columns are treated as categorical.
    - Nullable pandas integer/float dtypes are treated as numeric.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")

    boolean_cols = X_train.select_dtypes(include=["bool", "boolean"]).columns.tolist()

    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in boolean_cols]

    categorical_cols = X_train.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    recognized = set(numeric_cols) | set(categorical_cols) | set(boolean_cols)
    unrecognized = [c for c in X_train.columns if c not in recognized]

    if unrecognized:
        raise TypeError(
            "Unsupported dtypes found for columns: "
            f"{unrecognized}. Convert them before building the pipeline."
        )

    return FeatureGroups(
        numeric=numeric_cols,
        categorical=categorical_cols,
        boolean=boolean_cols,
    )


def _numeric_to_float(values):
    """Normalize pandas nullable numeric dtypes to regular float + np.nan."""
    if isinstance(values, pd.DataFrame):
        return values.astype(float)
    if isinstance(values, pd.Series):
        return values.astype(float)
    return np.asarray(values, dtype=float)


def _object_with_np_nan(values):
    """Convert pd.NA to np.nan so sklearn imputers can handle nullable dtypes."""
    if isinstance(values, pd.DataFrame):
        obj = values.astype(object)
        return obj.where(pd.notna(obj), np.nan)
    if isinstance(values, pd.Series):
        obj = values.astype(object)
        return obj.where(pd.notna(obj), np.nan)
    arr = np.asarray(values, dtype=object).copy()
    arr[pd.isna(arr)] = np.nan
    return arr


def _to_float_array(values):
    """Convert imputed boolean values to 0/1 float values."""
    return np.asarray(values, dtype=float)


def build_preprocessor(
    X_train: pd.DataFrame,
    *,
    scale_numeric: bool = False,
    categorical_min_frequency: Optional[int | float] = None,
) -> tuple[ColumnTransformer, FeatureGroups]:
    """Build an unfitted preprocessing pipeline using X_train schema only.

    Parameters
    ----------
    X_train:
        Training feature DataFrame. Used only to identify feature dtypes/columns.
        No statistics are learned in this function.
    scale_numeric:
        If True, numeric features are standardized after median imputation.
        Use for scale-sensitive models (e.g. linear/regularized models, KNN, SVR).
        Keep False for tree-based models unless there is another reason to scale.
    categorical_min_frequency:
        Optional rare-category threshold passed to OneHotEncoder. ``None`` keeps
        every observed category. This can be tuned later if high-cardinality
        categorical columns are retained by the feature-selection task.

    Returns
    -------
    preprocessor, groups
        An UNFITTED ColumnTransformer and the inferred feature groups.
    """
    groups = infer_feature_groups(X_train)

    numeric_steps = [
        (
            "normalize_nullable",
            FunctionTransformer(
                _numeric_to_float,
                validate=False,
                feature_names_out="one-to-one",
            ),
        ),
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            (
                "normalize_nullable",
                FunctionTransformer(
                    _object_with_np_nan,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__missing__",
                    keep_empty_features=True,
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    min_frequency=categorical_min_frequency,
                ),
            ),
        ]
    )

    boolean_pipeline = Pipeline(
        steps=[
            (
                "normalize_nullable",
                FunctionTransformer(
                    _object_with_np_nan,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "imputer",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            (
                "to_float",
                FunctionTransformer(
                    _to_float_array,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    transformers = []
    if groups.numeric:
        transformers.append(("numeric", numeric_pipeline, groups.numeric))
    if groups.categorical:
        transformers.append(("categorical", categorical_pipeline, groups.categorical))
    if groups.boolean:
        transformers.append(("boolean", boolean_pipeline, groups.boolean))

    if not transformers:
        raise ValueError("X_train contains no supported feature columns.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=True,
    )

    return preprocessor, groups


def fit_transform_feature_splits(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """Fit ONLY on train, then transform train/validation/test.

    This function encodes the central anti-leakage rule of Task 6:
    validation and test are never used in ``fit`` or ``fit_transform``.
    """
    expected_columns = list(X_train.columns)

    for split_name, split in {
        "X_validation": X_validation,
        "X_test": X_test,
    }.items():
        if list(split.columns) != expected_columns:
            raise ValueError(
                f"{split_name} columns/order do not match X_train. "
                "All feature splits must have identical columns."
            )

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_validation_transformed = preprocessor.transform(X_validation)
    X_test_transformed = preprocessor.transform(X_test)

    return X_train_transformed, X_validation_transformed, X_test_transformed


def build_model_pipeline(
    X_train: pd.DataFrame,
    estimator,
    *,
    scale_numeric: bool = False,
    categorical_min_frequency: Optional[int | float] = None,
) -> tuple[Pipeline, FeatureGroups]:
    """Combine preprocessing and an estimator in one leakage-safe sklearn Pipeline.

    Calling ``pipeline.fit(X_train, y_train)`` learns imputation, category
    vocabulary, scaling statistics, and model parameters from TRAIN only.
    ``predict`` on validation/test performs transform-only preprocessing.
    """
    preprocessor, groups = build_preprocessor(
        X_train,
        scale_numeric=scale_numeric,
        categorical_min_frequency=categorical_min_frequency,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    return pipeline, groups


def preprocessing_summary(X_train: pd.DataFrame) -> pd.DataFrame:
    """Return a compact summary useful for documenting Task 6."""
    groups = infer_feature_groups(X_train)

    rows = []
    for kind, columns in {
        "numeric": groups.numeric,
        "categorical": groups.categorical,
        "boolean": groups.boolean,
    }.items():
        for col in columns:
            rows.append(
                {
                    "feature": col,
                    "group": kind,
                    "dtype": str(X_train[col].dtype),
                    "missing_count": int(X_train[col].isna().sum()),
                    "missing_pct": float(X_train[col].isna().mean() * 100),
                    "n_unique": int(X_train[col].nunique(dropna=True)),
                }
            )

    return pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)
