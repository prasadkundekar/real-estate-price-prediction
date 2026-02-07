from fastapi import APIRouter, HTTPException
from app.api.schemas import HouseFeatures, PredictionResponse
from src.models.predict import predict_price

router = APIRouter()


@router.get("/")
def home():
    return {"message": "Real Estate Price Prediction API is running 🚀"}


@router.post("/predict", response_model=PredictionResponse)
def predict(data: HouseFeatures):
    try:
        price = predict_price(
            location=data.location,
            sqft=data.total_sqft,
            bath=data.bath,
            bhk=data.bhk,
        )

        return PredictionResponse(predicted_price=round(price, 2))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
