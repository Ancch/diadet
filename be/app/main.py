from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .schemas import DiabetesInput
from .predict import predict_diabetes

app = FastAPI()

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def home():
    return FileResponse("app/static/index.html")

@app.post("/predict")
def predict(input: DiabetesInput):
    pred, prob = predict_diabetes(input.dict())
    return {
        "diabetes": bool(pred),
        "probability": prob
    }
