# Chest X-Ray Pneumonia Classification

A deep learning project for automated pneumonia detection from chest X-ray images using TensorFlow and transfer learning.

## Overview

This project uses a pre-trained MobileNetV2 convolutional neural network to classify chest X-ray images into two categories:

* NORMAL
* PNEUMONIA

The objective is to build a complete machine learning pipeline, from data loading and training to model evaluation and deployment.

## Dataset

Dataset: Chest X-Ray Images (Pneumonia)

Structure:

```text
data/raw/chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

Training set distribution:

| Class     | Images |
| --------- | -----: |
| NORMAL    |  1,341 |
| PNEUMONIA |  3,875 |

The dataset is highly imbalanced, with approximately 74% pneumonia images.

## Model

### Architecture

* MobileNetV2 (ImageNet pre-trained)
* Transfer Learning
* GlobalAveragePooling2D
* Dropout
* Dense(1, sigmoid)

### Training

* TensorFlow / Keras
* Binary Cross-Entropy
* Adam Optimizer
* EarlyStopping
* ModelCheckpoint
* Class Weights for imbalance handling

## Results

### Training Curves

![Training Curves](images/training_curves.png)

### Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### ROC Curve

![ROC Curve](images/roc_curve.png)

### Classification Report

```text
              precision    recall  f1-score   support

NORMAL            0.96      0.60      0.74       234
PNEUMONIA         0.80      0.98      0.88       390

accuracy                              0.84       624
macro avg         0.88      0.79      0.81       624
weighted avg      0.86      0.84      0.83       624
```

### Observations

* The model achieves strong pneumonia detection performance.
* Recall for pneumonia is very high (98%).
* The dataset is significantly imbalanced.
* The model tends to over-predict the pneumonia class.

## Project Structure

```text
.
├── configs/
├── data/
├── images/
├── notebooks/
├── saved_models/
├── scripts/
└── src/
    ├── data/
    ├── models/
    ├── utils/
    ├── evaluate.py
    └── train.py
```

## Installation

```bash
git clone <repo-url>
cd xray_classification

pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

## Dataset Inspection

```bash
python -m scripts.inspect_dataset
```

Current exploration includes:

- class distribution
- sample visualization
- image dimensions
- dataset sanity checks

---

## Future Improvements

- Optimize threshold for binary classification.
- Analyze false positives and false negatives.
- Experiment with data augmentation strategies.
- Fine-tune the pre-trained MobileNetV2 layers.

## Next steps

- Build an inference pipeline for single-image predictions.
- Expose the model through a FastAPI service.
- Containerize the application with Docker.
- Deploy the service to AWS.
- Add automated testing and CI/CD workflows.

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* YAML
* Git
* GitHub
