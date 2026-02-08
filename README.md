# 🏠 Real Estate Price Prediction System

An end-to-end **Machine Learning web application** that predicts real estate prices in Bengaluru based on user inputs such as location, square footage, number of bathrooms, and BHK configuration.

This project demonstrates the **complete ML lifecycle** including data preprocessing, model training, backend API development, and an interactive dashboard.

---

## 🚀 Features

* 📊 Data cleaning, preprocessing, and feature engineering
* 🤖 Machine Learning price prediction model
* ⚡ FastAPI backend for real-time predictions
* 🖥️ Streamlit dashboard for user interaction
* 📈 SHAP explainability for feature impact analysis
* 📂 Organized, production-style project structure

---

## 🧠 Tech Stack

**Machine Learning**

* Python
* Pandas, NumPy
* Scikit-learn / LightGBM
* SHAP (Explainable AI)

**Backend**

* FastAPI
* Uvicorn

**Frontend**

* Streamlit
* Plotly

**Tools**

* Git & GitHub
* Virtual Environment

---

## 📁 Project Structure

```
REAL_ESTATE_PRICE_PREDICTION/
│
├── app/
│   ├── api/            # FastAPI routes and schemas
│   ├── dashboard/      # Streamlit UI
│   └── __init__.py
│
├── artifacts/models/   # Trained model and metadata
│   ├── model.pkl
│   ├── columns.json
│   └── locations.json
│
├── data/raw/           # Dataset
│   └── bengaluru_house_prices.csv
│
└── src/models/         # ML training & prediction logic
    ├── train.py
    ├── predict.py
    ├── preprocess.py
    └── shap_explainer.py
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/real-estate-price-prediction.git
cd real-estate-price-prediction
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python src/models/train.py
```

This will generate:

```
artifacts/models/model.pkl
artifacts/models/columns.json
artifacts/models/locations.json
```

---

## 🚀 Run the Application

### Start FastAPI backend

```bash
uvicorn app.api.main:app --reload
```

API Docs available at:

```
http://127.0.0.1:8000/docs
```

### Start Streamlit dashboard

```bash
streamlit run app/dashboard/app.py
```

---

## 📊 Example Prediction Inputs

* **Location:** Indira Nagar
* **Total Sqft:** 1000
* **Bathrooms:** 2
* **BHK:** 2

The system returns the **estimated property price** along with **feature impact visualization**.

---

## 🎯 Learning Outcomes

* Built a **full ML pipeline** from raw data to deployment
* Implemented **REST API for model serving**
* Created an **interactive analytical dashboard**
* Applied **Explainable AI (SHAP)** for transparency
* Followed **clean project architecture used in industry**

---

## 📌 Future Improvements

* Cloud deployment (AWS / Streamlit Cloud / Render)
* User authentication & history tracking
* Advanced ML models with higher R² score
* Real-time map-based visualization

---

## 👨‍💻 Author

**Prasad Kundekar**
B.Tech Computer Engineering Student
Aspiring **Data Scientist / ML Engineer**

---

## ⭐ If you like this project

Give it a **star on GitHub** and feel free to contribute!
