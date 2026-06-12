import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(config):

    image_size = config["data"]["image_size"]

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False
    
    # TODO:
    # MobileNetV2 was pretrained with preprocess_input()
    # Replace generic Rescaling(1./255) with:
    # tf.keras.applications.mobilenet_v2.preprocess_input
    # before benchmarking the final model.

    model = models.Sequential([
        layers.Rescaling(1./255),

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