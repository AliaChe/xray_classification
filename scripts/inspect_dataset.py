import tensorflow as tf
from src.data.load_data import load_datasets
from src.utils.load_config import load_config
import matplotlib.pyplot as plt

config = load_config() 

train_ds, val_ds, test_ds, class_names = load_datasets(config)

print("\n class names:", class_names)
print("\n train batches:", tf.data.experimental.cardinality(train_ds))
print("val batches:", tf.data.experimental.cardinality(val_ds))
print("test batches:", tf.data.experimental.cardinality(test_ds), "\n")

for images, labels in train_ds.take(1):

    print("images shape:", images.shape)
    print("labels shape:", labels.shape, "\n")

for images, labels in train_ds.take(1):

    plt.figure(figsize=(10, 10))

    for i in range(9):
        
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[int(labels[i])])

        plt.axis("off")

    plt.show()