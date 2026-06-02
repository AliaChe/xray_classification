from data.load_data import load_datasets
from utils.load_config import load_config

config = load_config() 

train_ds, val_ds = load_datasets(config)

for images, labels in train_ds.take(1):

    print(images.shape)
    print(labels.shape)

    print(labels[:10])

import matplotlib.pyplot as plt

class_names = train_ds.class_names

print(class_names)

for images, labels in train_ds.take(1):

    plt.figure(figsize=(10, 10))

    for i in range(9):
        
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))

        plt.title(class_names[int(labels[i])])

        plt.axis("off")

    plt.show()