from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import nest_asyncio

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

MODEL = tf.keras.models.load_model("../Models/1")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive.."

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))  # Replace with object
    return image


@app.post("/predict")
async def predict(
        file : UploadFile = File(...)
) :
    image = read_file_as_image ( await file.read())
    img_batch = np.expand_dims(image , 0)

    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'Class' : predicted_class,
        'Confidence' : float(confidence)*100
    }


nest_asyncio.apply()

if __name__ == "__main__" :
    uvicorn.run(app, host = 'localhost', port = 8000)
