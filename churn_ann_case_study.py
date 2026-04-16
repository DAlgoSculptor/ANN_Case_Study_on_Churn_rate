"""
Customer Churn Prediction with ANN (TensorFlow/Keras)

This script:
1) Loads the churn dataset
2) Preprocesses numeric and categorical features
3) Splits train/test data
4) Trains an ANN model
5) Evaluates and reports metrics
6) Saves artifacts (preprocessor + model)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow import keras


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    return pd.read_csv(csv_path)


def build_preprocessor(
    numerical_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric + categorical columns."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )


def prepare_features_and_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Prepare model input features and target.

    Drops identifier-like columns that do not contribute to predictive signal.
    """
    drop_columns = ["RowNumber", "CustomerId", "Surname", "Exited"]
    missing = [col for col in drop_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    x = df.drop(columns=drop_columns)
    y = df["Exited"].astype(int)

    categorical_features = ["Geography", "Gender"]
    numerical_features = [c for c in x.columns if c not in categorical_features]
    return x, y, numerical_features, categorical_features


def build_model(input_dim: int) -> keras.Model:
    """Define ANN architecture."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: pd.Series) -> dict:
    """Return dictionary of key classification metrics."""
    y_proba = model.predict(x_test, verbose=0).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n=== Evaluation Metrics (Test Set) ===")
    for key, value in metrics.items():
        print(f"{key:>10}: {value:.4f}")

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))
    return metrics


def main() -> None:
    data_path = Path("Artificial_Neural_Network_Case_Study_data.csv")
    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {data_path.resolve()}")
    df = load_data(data_path)
    print(f"Dataset shape: {df.shape}")

    x, y, numerical_features, categorical_features = prepare_features_and_target(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    preprocessor = build_preprocessor(numerical_features, categorical_features)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)

    if hasattr(x_train_processed, "toarray"):
        x_train_processed = x_train_processed.toarray()
        x_test_processed = x_test_processed.toarray()

    model = build_model(input_dim=x_train_processed.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]

    print("\nTraining ANN model...")
    history = model.fit(
        x_train_processed,
        y_train.values,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1,
    )

    evaluate_model(model, x_test_processed, y_test)

    model_path = output_dir / "churn_ann_model.keras"
    preprocessor_path = output_dir / "churn_preprocessor.joblib"
    history_path = output_dir / "training_history.joblib"

    model.save(model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(history.history, history_path)

    print("\nArtifacts saved:")
    print(f"- Model: {model_path.resolve()}")
    print(f"- Preprocessor: {preprocessor_path.resolve()}")
    print(f"- Training history: {history_path.resolve()}")


if __name__ == "__main__":
    main()
