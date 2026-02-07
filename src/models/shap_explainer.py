import shap
import json
import pickle
import numpy as np

MODEL_PATH = "artifacts/models/model.pkl"
COLUMNS_PATH = "artifacts/models/columns.json"

# ---------- Load model ----------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ---------- Load columns ----------
with open(COLUMNS_PATH, "r") as f:
    columns = json.load(f)["columns"]

# ---------- Create correct SHAP explainer ----------
# Works for sklearn linear / tree models
explainer = shap.Explainer(model, feature_names=columns)


def get_shap_values(location: str, sqft: float, bath: int, bhk: int):
    """
    Returns SHAP values for a single prediction
    """

    x = np.zeros(len(columns))

    x[columns.index("total_sqft")] = sqft
    x[columns.index("bath")] = bath
    x[columns.index("bhk")] = bhk

    loc_col = f"location_{location}"
    if loc_col in columns:
        x[columns.index(loc_col)] = 1

    # reshape for model input
    x = x.reshape(1, -1)

    # disable strict additivity check (safe for visualization)
    shap_values = explainer(x, check_additivity=False)

    return shap_values, columns
