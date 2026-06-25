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

## Training

```bash
git clone <repo-url>
cd xray_classification

pip install -r requirements/train.txt
```

```bash
python -m src.train
```
The trained model is saved to:

```text
saved_models/best_model.keras
```

This artifact is required for inference, FastAPI serving, and Docker deployment.

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

## Inference

The project includes a standalone inference pipeline for predicting pneumonia from a single chest X-ray image.

Run:

```bash
python -m src.predict path/to/image.jpeg
```

Example:

```bash
python -m src.predict data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg
```

Output:

```text
Prediction: PNEUMONIA
Confidence: 94.2%
Raw pneumonia score: 0.9421
```

### Inference Pipeline

```text
Chest X-Ray Image
        ↓
Resize (224 × 224)
        ↓
MobileNetV2 preprocess_input()
        ↓
Loaded Keras Model
        ↓
Sigmoid Probability
        ↓
NORMAL / PNEUMONIA
```

The inference pipeline uses the same preprocessing steps as training to ensure consistent predictions.

## Deployment

### FastAPI API

The project includes a FastAPI application exposing the trained model through a REST API.

Run locally:

```bash
uvicorn app.main:app --reload
```

Swagger documentation:

```text
http://localhost:8000/docs
```

### API Endpoints

#### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

#### Predict Pneumonia

```http
POST /predict
```

Upload a chest X-ray image and receive a prediction:

```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.94,
  "raw_score": 0.94
}
```

Input validation includes:

* image type validation
* empty file validation
* automatic temporary file cleanup

---

## Docker

Before building the Docker image, ensure that a trained model exists:

```text
saved_models/best_model.keras
```

If not, train the model first:

```bash
python -m src.train
```

Build the Docker image:

```bash
docker build -t xray-api .
```

Run the container:

```bash
docker run -p 8000:8000 xray-api
```

The API will be available at:

```text
http://localhost:8000/docs
```

---

## Docker Hub

The application image is published on Docker Hub and can be pulled directly:

```bash
docker pull <dockerhub-username>/xray-api:latest
```

Run:

```bash
docker run -p 8000:8000 <dockerhub-username>/xray-api:latest
```

## AWS Deployment

The application is containerized with Docker and deployed on an AWS EC2 instance.

The deployment workflow is:

Training
→ Saved Keras Model
→ Docker Image
→ Docker Hub
→ AWS EC2
→ FastAPI Service

The API is exposed through FastAPI and can be accessed via the public EC2 IP address.
![AWS Deployment](images/swagger_EC2.png)

## Continuous Integration

The project uses GitHub Actions to automatically validate every push.

The CI pipeline performs the following steps:

1. Set up Python 3.10
2. Install development dependencies
3. Run Ruff for static code analysis
4. Run Pytest
5. Build the Docker image
6. Push the image to Docker Hub

Only if all checks pass is the Docker image published.

## Future Improvements

- Perform detailed error analysis.
- Experiment with data augmentation strategies.
- Fine-tune the MobileNetV2 backbone.
- Adjust threshold for binary classification to improve model performance on the NORMAL class.

## Next steps

- Add CD workflow.
- Add model monitoring and logging.

## Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn
* YAML
* Git
* GitHub
