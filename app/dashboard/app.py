import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import json
import shap
import matplotlib.pyplot as plt
from src.models.shap_explainer import get_shap_values

API_URL = "https://real-estate-fastapi.onrender.com/predict"
COLUMNS_PATH = "artifacts/models/columns.json"


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# ---------- HIDE STREAMLIT DEFAULT UI ----------
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD LOCATIONS ----------
@st.cache_data
def load_locations():
    with open(COLUMNS_PATH, "r") as f:
        cols = json.load(f)["columns"]
    return sorted([c.replace("location_", "") for c in cols if "location_" in c])


locations = load_locations()


# ---------- HEADER ----------
st.markdown(
    "<h1 style='text-align:center;'>🏠 Real Estate Price Predictor</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center;color:gray;'>AI-powered property valuation system</p>",
    unsafe_allow_html=True,
)


# ---------- INPUT CARD ----------
with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("📍 Location", locations)
        sqft = st.number_input("📐 Total Sqft", 100, 10000, 1000)

    with col2:
        bath = st.number_input("🛁 Bathrooms", 1, 10, 2)
        bhk = st.number_input("🛏️ BHK", 1, 10, 2)

    predict_btn = st.button("🚀 Predict Price", use_container_width=True)


# ---------- FAST PREDICTION ----------
@st.cache_data(show_spinner=False)
def get_prediction(payload: dict):
    res = requests.post(API_URL, json=payload)

    if res.status_code != 200:
        raise ValueError("API request failed")

    data = res.json()

    if "predicted_price" not in data:
        raise ValueError("Invalid API response")

    return data["predicted_price"]


# ---------- MAIN ----------
if predict_btn:

    payload = {
        "location": location,
        "total_sqft": sqft,
        "bath": bath,
        "bhk": bhk,
    }

    with st.spinner("🔮 Predicting property price..."):

        try:
            price = get_prediction(payload)

            # ---------- PRICE DISPLAY ----------
            st.markdown("## 💰 Estimated Property Price")
            st.success(f"₹ {price:.2f} Lakhs")

            # ---------- SHAP PROFESSIONAL GRAPH ----------
            st.markdown("### 🔍 Feature Impact on Price")

            shap_values, columns = get_shap_values(location, sqft, bath, bhk)

            # Convert SHAP to dataframe for clean plotting
            shap_df = pd.DataFrame({
                "Feature": columns,
                "Impact": shap_values.values[0]
            })

            shap_df = shap_df.sort_values("Impact")

            fig_shap = px.bar(
                shap_df,
                x="Impact",
                y="Feature",
                orientation="h",
                title="How Each Feature Affects Price",
                color="Impact",
                color_continuous_scale="RdYlGn",
            )

            st.plotly_chart(fig_shap, use_container_width=True)

            # ---------- PRICE RANGE GRAPH ----------
            df_range = pd.DataFrame({
                "Range": ["Low", "Predicted", "High"],
                "Price": [price * 0.9, price, price * 1.1],
            })

            fig_range = px.bar(
                df_range,
                x="Range",
                y="Price",
                text_auto=".2f",
                color="Range",
                title="Estimated Price Range",
            )

            st.plotly_chart(fig_range, use_container_width=True)

            # ---------- PRICE TREND ----------
            sqft_values = list(range(500, 3000, 300))

            trend_prices = [
                get_prediction({
                    "location": location,
                    "total_sqft": s,
                    "bath": bath,
                    "bhk": bhk,
                })
                for s in sqft_values
            ]

            df_trend = pd.DataFrame({"Sqft": sqft_values, "Price": trend_prices})

            fig_trend = px.line(
                df_trend,
                x="Sqft",
                y="Price",
                markers=True,
                title="Price Trend vs Square Feet",
            )

            st.plotly_chart(fig_trend, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")
