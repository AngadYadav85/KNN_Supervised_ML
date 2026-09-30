"""Data loading, model training and evaluation for the loan-approval KNN app.

Mirrors the notebook pipeline:
    dropna -> drop applicant_id -> one-hot (drop first) -> 67/33 split
    (random_state=42) -> StandardScaler -> KNeighborsClassifier
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

TARGET = "loan_status"
ID_COL = "applicant_id"
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "monthly_emi",
    "credit_score",
    "existing_loans",
    "has_property",
]
CATEGORICAL_FEATURES = ["city", "education", "employment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]

TEST_SIZE = 0.33
RANDOM_STATE = 42


@dataclass
class TrainedModel:
    pipeline: Pipeline
    label_encoder: LabelEncoder
    k: int
    n_train: int
    n_test: int
    accuracy: float
    confusion: np.ndarray
    report: dict[str, Any]
    class_names: list[str]


def load_data(source) -> pd.DataFrame:
    """Read and clean the loan CSV. `source` is a path or file-like object."""
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required column(s): "
            + ", ".join(missing)
            + ". Expected columns: "
            + ", ".join(REQUIRED_COLUMNS)
            + f" (and optionally '{ID_COL}')."
        )

    if ID_COL in df.columns:
        df = df.drop(columns=ID_COL)
    df = df[REQUIRED_COLUMNS].dropna().reset_index(drop=True)

    for col in CATEGORICAL_FEATURES + [TARGET]:
        df[col] = df[col].astype(str).str.strip()

    if df[TARGET].nunique() < 2:
        raise ValueError(f"'{TARGET}' needs at least two classes to train a model.")
    if len(df) < 30:
        raise ValueError("Not enough rows after cleaning to train a model.")
    return df


def build_pipeline(k: int) -> Pipeline:
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", "passthrough", NUMERIC_FEATURES),
        ],
        sparse_threshold=0.0,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("scale", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
        ]
    )


def _split(df: pd.DataFrame):
    X = df[FEATURES]
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, le


def train_model(df: pd.DataFrame, k: int = 5) -> TrainedModel:
    X_train, X_test, y_train, y_test, le = _split(df)
    k = int(min(k, len(X_train)))
    pipe = build_pipeline(k).fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    labels = list(range(len(le.classes_)))
    return TrainedModel(
        pipeline=pipe,
        label_encoder=le,
        k=k,
        n_train=len(X_train),
        n_test=len(X_test),
        accuracy=float(accuracy_score(y_test, y_pred)),
        confusion=confusion_matrix(y_test, y_pred, labels=labels),
        report=classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=list(le.classes_),
            output_dict=True,
            zero_division=0,
        ),
        class_names=list(le.classes_),
    )


def predict_one(model: TrainedModel, row: dict[str, Any]) -> dict[str, Any]:
    """Predict for a single applicant. Returns label and per-class probabilities."""
    X = pd.DataFrame([row], columns=FEATURES)
    pipe = model.pipeline
    pred_idx = int(pipe.predict(X)[0])
    probs = pipe.predict_proba(X)[0]
    proba = {
        model.class_names[int(c)]: float(p) for c, p in zip(pipe.classes_, probs)
    }
    return {"label": model.class_names[pred_idx], "proba": proba}


def sweep_k(df: pd.DataFrame, ks: list[int]) -> pd.DataFrame:
    """Test accuracy / recall of the minority class for several values of k."""
    X_train, X_test, y_train, y_test, le = _split(df)
    minority = int(np.argmin(np.bincount(y_train)))
    rows = []
    for k in ks:
        if k > len(X_train):
            continue
        pipe = build_pipeline(k).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(
            {
                "k": k,
                "accuracy": accuracy_score(y_test, pred),
                f"recall ({le.classes_[minority]})": recall_score(
                    y_test, pred, labels=[minority], average=None, zero_division=0
                )[0],
            }
        )
    return pd.DataFrame(rows).set_index("k")
