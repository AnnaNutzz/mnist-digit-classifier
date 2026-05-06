import io
import os
import joblib
import numpy as np
from PIL import Image, ImageOps
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
MODEL_PATH = "model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    print(f"Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
else:
    print(f"Model file {MODEL_PATH} not found. Please run model_train.py first.")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Please train the model first."}
    
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Preprocess the image
        # 1. Convert to grayscale
        image = ImageOps.grayscale(image)
        # 2. Resize to 28x28 (standard MNIST size)
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        # 3. Invert colors if necessary (MNIST is white digits on black background)
        # We'll assume the user draws black on white, so we invert.
        # But wait, let's check the average pixel value.
        # If it's mostly light, invert it.
        img_array = np.array(image)
        if np.mean(img_array) > 127:
            image = ImageOps.invert(image)
        
        # 4. Normalize (0-255 -> 0-1)
        data = np.array(image).astype(np.float32) / 255.0
        # 5. Flatten to 784 features
        data = data.reshape(1, -1)
        
        # Predict
        prediction = model.predict(data)[0]
        # Get probabilities if available
        probabilities = model.predict_proba(data)[0].tolist()
        
        return {
            "prediction": str(prediction),
            "confidence": float(probabilities[int(prediction)]),
            "probabilities": {str(i): float(probabilities[i]) for i in range(10)}
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Use port 3000 as it is the only externally accessible port in AI Studio
    port = int(os.environ.get("PORT", 3000))
    print(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
