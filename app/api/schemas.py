from pydantic import BaseModel


class HouseFeatures(BaseModel):
    location: str
    total_sqft: float
    bath: int
    bhk: int


class PredictionResponse(BaseModel):
    predicted_price: float
