import subprocess
from pathlib import Path

import mlflow
import mlflow.tensorflow


def log_git_commit():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()

        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            text=True,
        ).strip()

        mlflow.set_tag("git_branch", branch)
        mlflow.set_tag("git_commit", commit)

    except Exception:
        pass


def log_params(config):
    params = {
        **config["training"],
        "image_size": config["data"]["image_size"],
        "dropout": config["model"]["dropout"],
        "architecture": config["model"]["architecture"],
        "threshold": config["evaluation"]["threshold"],
    }

    mlflow.log_params(params)
    mlflow.log_artifact("configs/config.yaml")


def set_tags():
    mlflow.set_tags(
        {
            "framework": "TensorFlow",
            "model": "MobileNetV2",
            "dataset": "Chest X-Ray Pneumonia",
            "stage": "baseline",
            "author": "Alia",
        }
    )


def log_history(history):
    for metric_name, values in history.history.items():
        for epoch, value in enumerate(values):
            mlflow.log_metric(metric_name, value, step=epoch)


def log_test_metrics(test_loss, test_acc):
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_acc)


def log_artifacts(artifacts_dir):
    for artifact in Path(artifacts_dir).rglob("*"):
        if artifact.is_file():
            mlflow.log_artifact(str(artifact))


def log_model(model):
    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
    )
