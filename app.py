from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from predict import load_all_models, predict_image
from PIL import Image
import io

app = FastAPI(title="Screen vs Camera Detector API")

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Replace ["*"] with a list of allowed domains for better security.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

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
