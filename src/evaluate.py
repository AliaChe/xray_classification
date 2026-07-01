import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve,
    auc,
)


def plot_training_curves(history, artifacts_dir):
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training / Validation Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training / Validation Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/training_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrix(model, test_ds, class_names, artifacts_dir, threshold=0.5):

    y_true, y_scores = get_predictions(model, test_ds)

    y_pred = (y_scores > threshold).astype(int)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
    )

    (artifacts_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    _, ax = plt.subplots(figsize=(6, 6))

    disp.plot(ax=ax)

    plt.title(f"Confusion Matrix (threshold={threshold})")
    plt.savefig(f"{artifacts_dir}/confusion_matrix.png")
    plt.show()


def get_predictions(model, test_ds):

    y_true = []
    y_scores = []

    for images, labels in test_ds:
        scores = model.predict(images, verbose=0).flatten()

        y_true.extend(labels.numpy())
        y_scores.extend(scores)

    return (np.array(y_true), np.array(y_scores))


def plot_roc_curve(model, test_ds, artifacts_dir):
    y_true, y_scores = get_predictions(model, test_ds)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{artifacts_dir}/roc_curve.png", dpi=150)
    plt.show()

    print(f"ROC AUC: {roc_auc:.4f}")

    return fpr, tpr, thresholds
