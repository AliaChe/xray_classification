import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from src.data.load_data import load_raw_datasets
from src.utils.load_config import load_config

def print_dataset_info(train_ds, val_ds, test_ds, class_names):
    print("\n=== Dataset Information ===")

    print(f"Classes: {class_names}")

    print(f"Train batches: {tf.data.experimental.cardinality(train_ds).numpy()}")
    print(f"Val batches:   {tf.data.experimental.cardinality(val_ds).numpy()}")
    print(f"Test batches:  {tf.data.experimental.cardinality(test_ds).numpy()}")


def inspect_batch(train_ds):
    print("\n=== Sample Batch ===")

    images, labels = next(iter(train_ds))

    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")

    return images, labels


def plot_samples(images, labels, class_names, n_samples=9):
    plt.figure(figsize=(10, 10))

    for i in range(min(n_samples, len(images))):
        plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def analyze_class_distribution(train_path):
    print("\n=== Class Distribution ===")

    counts = Counter()

    for class_dir in Path(train_path).iterdir():
        if class_dir.is_dir():
            counts[class_dir.name] = len(list(class_dir.glob("*")))

    total = sum(counts.values())

    for class_name, count in counts.items():
        print(
            f"{class_name}: "
            f"{count} images "
            f"({count / total:.2%})"
        )

    return counts


def plot_class_distribution(counts):
    plt.figure(figsize=(6, 4))

    plt.bar(
        counts.keys(),
        counts.values()
    )

    plt.title("Class Distribution")
    plt.ylabel("Number of Images")

    plt.tight_layout()
    plt.show()


def main():
    config = load_config()

    train_ds, val_ds, test_ds, class_names = load_raw_datasets(config)

    print_dataset_info(
        train_ds,
        val_ds,
        test_ds,
        class_names
    )

    images, labels = inspect_batch(train_ds)

    plot_samples(
        images,
        labels,
        class_names
    )

    counts = analyze_class_distribution(
        config["data"]["path"]
    )

    plot_class_distribution(counts)


if __name__ == "__main__":
    main()