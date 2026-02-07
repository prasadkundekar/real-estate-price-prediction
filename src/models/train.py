import os
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from preprocess import load_and_clean_data


DATA_PATH = "data/raw/bengaluru_house_prices.csv"
MODEL_DIR = "artifacts/models"


def train():
    df = load_and_clean_data(DATA_PATH)

    df = df.drop(['size', 'price_per_sqft'], axis=1)

    X = pd.get_dummies(df.drop('price', axis=1), drop_first=True)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "linear_regression": LinearRegression(),
        "lasso": Lasso(),
        "decision_tree": DecisionTreeRegressor(),
        "random_forest": RandomForestRegressor()
    }

    params = {
        "linear_regression": {},
        "lasso": {"alpha": [0.1, 1, 10]},
        "decision_tree": {"max_depth": [5, 10, 20]},
        "random_forest": {"n_estimators": [50, 100]}
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        gs = GridSearchCV(model, params[name], cv=5)
        gs.fit(X_train, y_train)

        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_model = gs.best_estimator_

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    with open(os.path.join(MODEL_DIR, "columns.json"), "w") as f:
        json.dump({"columns": list(X.columns)}, f)

    print("✅ Training complete")
    print("Best CV Score:", best_score)


if __name__ == "__main__":
    train()
