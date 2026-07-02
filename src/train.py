from pathlib import Path

import mlflow

from src.data.load_data import compute_weights, load_datasets
from src.evaluate import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
)
from src.models.callbacks import build_callbacks
from src.models.cnn_model import build_model
from src.utils.load_config import load_config
from src.utils.mlflow_logging import (
    log_artifacts,
    log_git_commit,
    log_history,
    log_model,
    log_params,
    log_test_metrics,
    set_tags,
)

config = load_config()

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

mlflow.set_experiment(config["mlflow"]["experiment"])

train_ds, val_ds, test_ds, class_names = load_datasets(config)

class_weights = compute_weights(
    config["data"]["path"],
    class_names,
)

model = build_model(config)
model.summary()

training_callbacks = build_callbacks(config)

artifacts_dir = Path(config["paths"]["artifacts"])
artifacts_dir.mkdir(parents=True, exist_ok=True)

with mlflow.start_run():
    log_git_commit()
    log_params(config)
    set_tags()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        class_weight=class_weights,
        callbacks=training_callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds)

    plot_training_curves(history, artifacts_dir)
    plot_confusion_matrix(
        model,
        test_ds,
        class_names,
        artifacts_dir,
        config["evaluation"]["threshold"],
    )
    plot_roc_curve(model, test_ds, artifacts_dir)

    log_history(history)
    log_test_metrics(test_loss, test_acc)

    log_artifacts(artifacts_dir)

    log_model(model)

    model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name="ChestXRayClassifier",
    )

print(f"Test accuracy: {test_acc:.4f}")
