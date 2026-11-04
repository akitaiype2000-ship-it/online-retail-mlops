from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
import joblib
print(">>> LOADING app.py <<<")
app = FastAPI(
    title="Online Retail High Value Customer Prediction API"
)
instrumentator = Instrumentator()
instrumentator.instrument(app)
instrumentator.expose(app, endpoint="/metrics")
# Load model
model = joblib.load("artifacts/models/random_forest_model.pkl")


class CustomerData(BaseModel):
    total_revenue: float
    number_of_purchases: int
    purchase_frequency: int
    average_order_value: float
    recency_days: int
@app.get("/hello")
def hello():
    return {"message": "Hello"}

@app.get("/")
def home():
    return {"message": "API is running successfully!"}


@app.post("/predict")
def predict(data: CustomerData):

    features = [[
        data.total_revenue,
        data.number_of_purchases,
        data.purchase_frequency,
        data.average_order_value,
        data.recency_days
    ]]

    prediction = model.predict(features)[0]

    return {
        "prediction": int(prediction),
        "result": (
            "High Value Repeat Customer"
            if prediction == 1
            else "Not High Value Repeat Customer"
        )
    }