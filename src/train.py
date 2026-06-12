from data.load_data import load_datasets
from models.cnn_model import build_model
from utils.load_config import load_config

config = load_config()

train_ds, val_ds, test_ds, class_names = load_datasets(config)

model = build_model(config)

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=config["training"]["epochs"]
)