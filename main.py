import tensorflow as tf
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware 
from io import BytesIO
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

def preprocess_image_from_bytes(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((224, 224))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

try:
    model = tf.keras.models.load_model('models/weed_model_efficientnet-3.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict_image(image_bytes):
    img_array = preprocess_image_from_bytes(image_bytes)
    if img_array is None:
        return {"error": "Failed to preprocess image."}

    try:
        predictions = model.predict(img_array)[0]
        labels = ["daisy", "dandelion", "nutsedge", "potato", "purslane"]
        return {label: float(f"{pred:.4f}") for label, pred in zip(labels, predictions)}
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        prediction = predict_image(contents)
        return JSONResponse(content=prediction)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
