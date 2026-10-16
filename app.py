
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = FastAPI(title="Churn Prediction API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/predictdata", response_class=HTMLResponse)
async def predict_datapoint_get(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={"results": None, "error": None}
    )


@app.post("/predictdata", response_class=HTMLResponse)
async def predict_datapoint_post(
    request: Request,
    CreditScore: int = Form(..., ge=300, le=850),
    Geography: str = Form(...),
    Gender: str = Form(...),
    Age: int = Form(..., ge=18, le=100),
    Tenure: int = Form(..., ge=0, le=10),
    Balance: float = Form(..., ge=0)
):
    try:
        data = CustomData(
            CreditScore=CreditScore,
            Geography=Geography,
            Gender=Gender,
            Age=Age,
            Tenure=Tenure,
            Balance=Balance
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        prediction = int(results[0])
        prediction_label = "Yes" if prediction == 1 else "No"

        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={
                "results": prediction_label,
                "error": None
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="home.html",
            context={
                "results": None,
                "error": str(e)
            },
            status_code=500
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
