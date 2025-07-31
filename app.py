from fastapi import FastAPI, UploadFile, File
from predict import load_all_models, predict_image
from PIL import Image
import io

app = FastAPI(title="Screen vs Camera Detector API")

# Load models once at startup
model_info = load_all_models()

@app.get("/")
def root():
    return {"message": "Screen vs Camera Detection API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        result = predict_image(image, model_info)
        return result
    except Exception as e:
        return {"error": str(e)}
