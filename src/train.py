from tensorflow.keras import callbacks
import mlflow
from pathlib import Path
from src.data.load_data import load_datasets
from src.models.cnn_model import build_model
from src.utils.load_config import load_config
from src.data.load_data import compute_weights
from src.evaluate import plot_training_curves, plot_confusion_matrix, plot_roc_curve
from src.utils.mlflow_logging import log_git_commit

config = load_config()

mlflow.set_experiment("xray-classification")

train_ds, val_ds, test_ds, class_names = load_datasets(config)

class_weights = compute_weights(config["data"]["path"], class_names)
print(class_weights)

model = build_model(config)

model.summary()

training_callbacks = [
    callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config["training"]["patience"],
        restore_best_weights=True,
    ),
    callbacks.ModelCheckpoint(
        filepath="saved_models/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
    ),
]

with mlflow.start_run():

    params = {
        **config["training"],
        "image_size": config["data"]["image_size"],
        "architecture": config["model"]["architecture"]
    }

    log_git_commit()
    mlflow.log_params(params)
    mlflow.log_artifact("configs/config.yaml")

    mlflow.set_tags(
        {
            "framework": "TensorFlow",
            "model": "MobileNetV2",
            "dataset": "Chest X-Ray Pneumonia",
            "stage": "baseline",
            "author":"Alia"
        }
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        class_weight=class_weights,
        callbacks=training_callbacks,
    )

    test_loss, test_acc = model.evaluate(test_ds)

    print("Test accuracy:", test_acc)

    artifacts = config["paths"]["artifacts"]
    Path(artifacts).mkdir(exist_ok=True)

    plot_training_curves(history, artifacts)
    plot_confusion_matrix(model, test_ds, class_names, artifacts, threshold=0.5)
    plot_roc_curve(model, test_ds, artifacts)

    for metric_name, values in history.history.items():
        for epoch, value in enumerate(values):
            mlflow.log_metric(metric_name, value, step=epoch)
    
    for artifact in Path(artifacts).glob("*.png"):
        mlflow.log_artifact(str(artifact))

    mlflow.tensorflow.log_model(
        model=model,
        artifact_path="model",
    )
