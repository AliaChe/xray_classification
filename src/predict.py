import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def load_image(image_path, image_size):
    img = image.load_img(
        image_path,
        target_size=(image_size, image_size)
    )

    img_array = image.img_to_array(img)

    # same preprocessing as training
    img_array = preprocess_input(img_array)

    # add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict_image(model_path, image_path, image_size=224, threshold=0.5):
    model = tf.keras.models.load_model(model_path)

    img_array = load_image(image_path, image_size)

    score = model.predict(img_array, verbose=0)[0][0]

    predicted_label = "PNEUMONIA" if score > threshold else "NORMAL"

    confidence = score if predicted_label == "PNEUMONIA" else 1 - score

    return predicted_label, confidence, score


def main():
    parser = argparse.ArgumentParser(
        description="Predict pneumonia from a chest X-ray image."
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the chest X-ray image."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="saved_models/best_model.keras",
        help="Path to the trained Keras model."
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size."
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for pneumonia class."
    )

    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    label, confidence, raw_score = predict_image(
        model_path=args.model_path,
        image_path=image_path,
        image_size=args.image_size,
        threshold=args.threshold
    )

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Raw pneumonia score: {raw_score:.4f}")


if __name__ == "__main__":
    main()