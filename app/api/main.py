from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Real Estate Price Prediction API",
    description="Production-ready ML API using FastAPI",
    version="1.0.0",
)

app.include_router(router)
