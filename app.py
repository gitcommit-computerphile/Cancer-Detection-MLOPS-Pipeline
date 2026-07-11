import os
import sys
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from cancer.exception.exception import CancerException
from cancer.logging.logger import logger
from cancer.pipeline.training_pipeline import TrainingPipeline
from cancer.pipeline.batch_prediction import BatchPredictionPipeline

app = FastAPI(title="Breast Cancer Detection API")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("table.html", {"request": request, "results": None})


@app.get("/train")
def train():
    try:
        pipeline = TrainingPipeline()
        artifact = pipeline.run_pipeline()
        return {
            "status": "Training complete",
            "train_f1": round(artifact.train_metric_artifact.f1_score, 4),
            "test_f1": round(artifact.test_metric_artifact.f1_score, 4),
        }
    except Exception as e:
        raise CancerException(e, sys)


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        tmp_path = "tmp_upload.csv"
        with open(tmp_path, "wb") as f:
            f.write(await file.read())
        predictor = BatchPredictionPipeline(tmp_path)
        df = predictor.run_prediction()
        os.remove(tmp_path)
        results = df.to_dict(orient="records")
        return templates.TemplateResponse("table.html", {"request": request, "results": results})
    except Exception as e:
        raise CancerException(e, sys)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)