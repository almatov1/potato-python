import tensorflow as tf
from PIL import Image
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

# FastAPI
app = FastAPI()

# Image process
def preprocess_image_from_bytes(image_bytes):
    try:
        img = Image.open(image_bytes).convert("RGB")
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        return None

# Model loading
try:
    model = tf.keras.models.load_model('models/model.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Predict
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

# Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        prediction = predict_image(file.file)
        return JSONResponse(content=prediction)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Start
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
