from pathlib import Path
from mlflow import tensorflow as tf
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
from pydantic import BaseModel
from src.predict import predict_image
from src.utils.load_config import load_config


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    raw_score: float


app = FastAPI()

model = None
config = load_config()

threshold = config["evaluation"]["threshold"]
image_size = config["data"]["image_size"]


@app.on_event("startup")
def load_model():
    global model
    model = tf.load_model(
        "models:/ChestXRayClassifier@champion"
    )


Path("tmp").mkdir(exist_ok=True)


@app.get("/")
def root():
    return {"message": "Chest X-Ray Classification API"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty file.")

    temp_path = Path("tmp") / file.filename

    with open(temp_path, "wb") as buffer:
        buffer.write(contents)

    try:
        result = predict_image(
            model=model,
            image_path=temp_path,
            image_size=image_size,
            threshold=threshold,
        )

        return result

    finally:
        if temp_path.exists():
            temp_path.unlink()
