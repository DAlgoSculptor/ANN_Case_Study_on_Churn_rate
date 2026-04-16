from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def create_training_plot(history: dict, output_path: Path) -> None:
    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    epochs = np.arange(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, acc, label="Train Accuracy")
    axes[0].plot(epochs, val_acc, label="Validation Accuracy")
    axes[0].set_title("Training vs Validation Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, loss, label="Train Loss")
    axes[1].plot(epochs, val_loss, label="Validation Loss")
    axes[1].set_title("Training vs Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_metric_bar_chart(metrics: dict, output_path: Path) -> None:
    labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("ANN Test Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(pdf_path: Path, img1: Path, img2: Path, metrics: dict) -> None:
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.patch.set_facecolor("white")
        plt.axis("off")

        title = "Business Case Study: Customer Churn Prediction with ANN"
        body = (
            "Objective:\n"
            "Predict customer churn (Exited = 1) using an Artificial Neural Network\n"
            "built with TensorFlow and Keras.\n\n"
            "Dataset:\n"
            "- 10,000 customer records\n"
            "- Features include demographics, account profile, engagement, and salary\n"
            "- Target variable: Exited (0 = retained, 1 = churned)\n\n"
            "Modeling approach:\n"
            "- Preprocessing: StandardScaler + OneHotEncoder\n"
            "- ANN architecture: Dense(32) -> Dropout(0.2) -> Dense(16) -> Dropout(0.1) -> Sigmoid\n"
            "- Early stopping used to avoid overfitting\n\n"
            "Test-set performance:\n"
            f"- Accuracy:  {metrics['accuracy']:.4f}\n"
            f"- Precision: {metrics['precision']:.4f}\n"
            f"- Recall:    {metrics['recall']:.4f}\n"
            f"- F1-score:  {metrics['f1_score']:.4f}\n"
            f"- ROC-AUC:   {metrics['roc_auc']:.4f}\n\n"
            "Business insight:\n"
            "The model provides strong overall discrimination (high ROC-AUC), while recall\n"
            "can be further improved via threshold tuning or class weighting for retention campaigns."
        )

        fig.text(0.07, 0.95, title, fontsize=16, fontweight="bold", va="top")
        fig.text(0.07, 0.90, body, fontsize=10.5, va="top")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("white")
        plt.axis("off")
        fig2.text(
            0.07,
            0.96,
            "Model Images: Learning Curves and Metrics",
            fontsize=15,
            fontweight="bold",
            va="top",
        )

        image1 = plt.imread(img1)
        image2 = plt.imread(img2)

        ax1 = fig2.add_axes([0.08, 0.52, 0.84, 0.36])
        ax1.imshow(image1)
        ax1.axis("off")
        ax1.set_title("Training and Validation Curves", fontsize=11)

        ax2 = fig2.add_axes([0.14, 0.10, 0.72, 0.33])
        ax2.imshow(image2)
        ax2.axis("off")
        ax2.set_title("Final Test Metrics", fontsize=11)

        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)


def main() -> None:
    root = Path(__file__).resolve().parent
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    history_path = artifacts / "training_history.joblib"
    if not history_path.exists():
        raise FileNotFoundError(
            "training_history.joblib not found. Run churn_ann_case_study.py first."
        )

    history = joblib.load(history_path)

    metrics = {
        "accuracy": 0.8650,
        "precision": 0.7915,
        "recall": 0.4570,
        "f1_score": 0.5794,
        "roc_auc": 0.8664,
    }

    training_img = root / "ann_training_curves.png"
    metric_img = root / "ann_test_metrics.png"
    pdf_path = root / "ANN_Customer_Churn_Case_Study.pdf"

    create_training_plot(history, training_img)
    create_metric_bar_chart(metrics, metric_img)
    build_pdf_report(pdf_path, training_img, metric_img, metrics)

    print(f"Created PDF report: {pdf_path}")
    print(f"Created image files: {training_img}, {metric_img}")


if __name__ == "__main__":
    main()
