import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report

def plot_history(history):

    plt.figure(figsize=(8,5))

    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.legend()
    plt.savefig("images/training_curve.png")
    plt.show()


def plot_confusion_matrix(model, test_ds, class_names):

    y_true = []
    y_pred = []

    for images, labels in test_ds:

        predictions = model.predict(images, verbose=0)

        predictions = (predictions > 0.5).astype(int).flatten()

        y_true.extend(labels.numpy())
        y_pred.extend(predictions)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    _, ax = plt.subplots(figsize=(6,6))

    disp.plot(ax=ax)

    plt.title("Confusion Matrix")
    plt.savefig("images/confusion_matrix.png")
    plt.show()

