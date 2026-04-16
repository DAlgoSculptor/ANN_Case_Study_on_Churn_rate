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


def add_page_header(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.07, 0.96, title, fontsize=16, fontweight="bold", va="top")
    fig.text(0.07, 0.93, subtitle, fontsize=10.5, color="#444444", va="top")
    line = plt.Line2D([0.07, 0.93], [0.915, 0.915], color="#999999", linewidth=0.8)
    fig.add_artist(line)


def build_pdf_report(pdf_path: Path, img1: Path, img2: Path, metrics: dict) -> None:
    with PdfPages(pdf_path) as pdf:
        # Page 1: Executive report page
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        plt.axis("off")

        add_page_header(
            fig,
            "Customer Churn Prediction Case Study",
            "Artificial Neural Network (TensorFlow/Keras) | Professional Summary",
        )

        body = (
            "1. Problem Statement\n"
            "This study predicts churn probability for bank customers to support proactive\n"
            "retention interventions and reduce avoidable attrition.\n\n"
            "2. Dataset and Scope\n"
            "- 10,000 customer records\n"
            "- Target: Exited (0 = retained, 1 = churned)\n"
            "- Feature groups: demographics, account profile, engagement, salary\n"
            "- Identifier fields excluded: RowNumber, CustomerId, Surname\n\n"
            "3. Modeling Approach\n"
            "- Preprocessing: StandardScaler + OneHotEncoder\n"
            "- Train/test split: 80/20 with stratification\n"
            "- ANN: Dense(32) -> Dropout(0.20) -> Dense(16) -> Dropout(0.10) -> Sigmoid\n"
            "- Training controls: Adam optimizer, binary cross-entropy, early stopping\n\n"
            "4. Performance on Test Set\n"
            f"- Accuracy : {metrics['accuracy']:.4f}\n"
            f"- Precision: {metrics['precision']:.4f}\n"
            f"- Recall   : {metrics['recall']:.4f}\n"
            f"- F1-Score : {metrics['f1_score']:.4f}\n"
            f"- ROC-AUC  : {metrics['roc_auc']:.4f}\n\n"
            "5. Business Interpretation\n"
            "- Strong ROC-AUC indicates effective ranking of churn risk.\n"
            "- Precision is favorable for focused retention spend.\n"
            "- Recall can be improved by threshold optimization or class weighting.\n\n"
            "6. Recommended Next Steps\n"
            "- Calibrate decision thresholds by campaign budget and expected ROI.\n"
            "- Compare with gradient boosting baselines.\n"
            "- Add explainability (SHAP) for action-oriented interventions."
        )

        fig.text(0.07, 0.88, body, fontsize=10.6, va="top", linespacing=1.35)
        fig.text(
            0.07,
            0.05,
            "Prepared for: ANN Customer Churn Business Case Study",
            fontsize=9,
            color="#666666",
        )
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Visual evidence page
        fig2 = plt.figure(figsize=(8.27, 11.69))
        fig2.patch.set_facecolor("white")
        plt.axis("off")
        add_page_header(
            fig2,
            "Model Performance Visuals",
            "Training behavior and evaluation summary",
        )

        image1 = plt.imread(img1)
        image2 = plt.imread(img2)

        ax1 = fig2.add_axes([0.08, 0.53, 0.84, 0.35])
        ax1.imshow(image1)
        ax1.axis("off")
        ax1.set_title("Figure 1. Training and Validation Curves", fontsize=11, pad=6)

        ax2 = fig2.add_axes([0.14, 0.12, 0.72, 0.30])
        ax2.imshow(image2)
        ax2.axis("off")
        ax2.set_title("Figure 2. Final Test Metrics", fontsize=11, pad=6)

        fig2.text(
            0.08,
            0.06,
            "Note: Final deployment threshold should be selected based on retention cost and "
            "campaign capacity constraints.",
            fontsize=9,
            color="#555555",
        )
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
