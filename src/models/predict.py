import json
import pickle
import numpy as np


MODEL_PATH = "artifacts/models/model.pkl"
COLUMNS_PATH = "artifacts/models/columns.json"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(COLUMNS_PATH, "r") as f:
    columns = json.load(f)["columns"]


def predict_price(location: str, sqft: float, bath: int, bhk: int):
    x = np.zeros(len(columns))

    x[columns.index("total_sqft")] = sqft
    x[columns.index("bath")] = bath
    x[columns.index("bhk")] = bhk

    loc_col = f"location_{location}"
    if loc_col in columns:
        x[columns.index(loc_col)] = 1

    return float(model.predict([x])[0])
