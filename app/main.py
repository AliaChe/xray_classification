from pathlib import Path

import tensorflow as tf

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from src.predict import predict_image

app = FastAPI()

model = tf.keras.models.load_model(
    "saved_models/best_model.keras"
)

Path("tmp").mkdir(exist_ok=True)

@app.get("/")
def root():
    return {
        "message": "Chest X-Ray API"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image."
        )

    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file."
        )

    temp_path = Path("tmp") / file.filename

    with open(temp_path, "wb") as buffer:
        buffer.write(contents)

    try:

        result = predict_image(
            model=model,
            image_path=temp_path
        )
        
        return result
    
    finally:

        if temp_path.exists():
            temp_path.unlink()
