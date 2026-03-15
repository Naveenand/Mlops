from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

from heart_failure.constants import APP_HOST, APP_PORT
from heart_failure.pipeline.prediction_pipeline import HeartData, HeartClassifier
from heart_failure.pipeline.training_pipeline import TrainPipeline

app = FastAPI()

# Mount static folder for CSS/JS
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory='templates')

# CORS settings
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Form class to capture input from HTML
class HeartDataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.age: Optional[str] = None
        self.anaemia: Optional[str] = None
        self.creatinine_phosphokinase: Optional[str] = None
        self.diabetes: Optional[str] = None
        self.ejection_fraction: Optional[str] = None
        self.high_blood_pressure: Optional[str] = None
        self.platelets: Optional[str] = None
        self.serum_creatinine: Optional[str] = None
        self.serum_sodium: Optional[str] = None
        self.sex: Optional[str] = None
        self.smoking: Optional[str] = None
        self.time: Optional[str] = None

    async def get_heart_data(self):
        form = await self.request.form()
        self.age = form.get("age")
        self.anaemia = form.get("anaemia")
        self.creatinine_phosphokinase = form.get("creatinine_phosphokinase")
        self.diabetes = form.get("diabetes")
        self.ejection_fraction = form.get("ejection_fraction")
        self.high_blood_pressure = form.get("high_blood_pressure")
        self.platelets = form.get("platelets")
        self.serum_creatinine = form.get("serum_creatinine")
        self.serum_sodium = form.get("serum_sodium")
        self.sex = form.get("sex")
        self.smoking = form.get("smoking")
        self.time = form.get("time")

# Home route
@app.get("/", tags=["home"])
async def index(request: Request):
    return templates.TemplateResponse(
        "heart_failure.html", {"request": request, "context": "Rendering"}
    )

# Train route
@app.get("/train")
async def train_route():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Heart Failure Model Training Successful!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

# Prediction route
@app.post("/")
async def predict_route(request: Request):
    try:
        form = HeartDataForm(request)
        await form.get_heart_data()
        
        heart_data = HeartData(
            age=form.age,
            anaemia=form.anaemia,
            creatinine_phosphokinase=form.creatinine_phosphokinase,
            diabetes=form.diabetes,
            ejection_fraction=form.ejection_fraction,
            high_blood_pressure=form.high_blood_pressure,
            platelets=form.platelets,
            serum_creatinine=form.serum_creatinine,
            serum_sodium=form.serum_sodium,
            sex=form.sex,
            smoking=form.smoking,
            time=form.time
        )
        
        heart_df = heart_data.get_input_data_frame()
        model_predictor = HeartClassifier()
        value = model_predictor.predict(dataframe=heart_df)[0]

        status = "High Risk of Death" if value == 1 else "Low Risk of Death"

        return templates.TemplateResponse(
            "heart_failure.html",
            {"request": request, "context": status},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)