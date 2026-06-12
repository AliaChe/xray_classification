import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_model(config):

    image_size = config["data"]["image_size"]

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        layers.Lambda(preprocess_input),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.2),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"]
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model