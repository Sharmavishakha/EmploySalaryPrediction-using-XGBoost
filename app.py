
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error

st.title("💼 Salary Predictor")
st.markdown("Predict an employee's salary based on profile details. Upload company data to customize predictions.")

# Default model (trained on general data)
general_model = joblib.load("xgboost_salary_predictor.pkl")
custom_model = None
use_custom = False
model_scores = {}

# --- Company Data Upload Section ---
# --- Company Data Upload Section ---
st.subheader("📂 Upload Company Data")

with st.expander("ℹ️ Required CSV Format (click to view)"):
    st.markdown('''
    Your CSV must contain **exactly these 6 columns**:
    
    | Column Name            | Description                      |
    |------------------------|----------------------------------|
    | `Age`                  | Age of the employee (numeric)    |
    | `Gender`               | `Male` or `Female`               |
    | `Education Level`      | `Bachelor's`, `Master's`, or `PhD` |
    | `Job Title`            | One of the following:<br>`Software Engineer`, `Data Analyst`, `Senior Manager`, `Sales Associate`, `Director`, `Marketing Analyst` |
    | `Years of Experience`  | Numeric experience in years      |
    | `Salary`               | Current salary (numeric)         |
    
    **Note**: Column names must match exactly, and categorical values must follow the given options.
    ''')

uploaded_file = st.file_uploader("📁 Upload company-specific employee data (CSV)", type=["csv"])

if uploaded_file:
    company_df = pd.read_csv(uploaded_file)
    st.success("✅ Company data uploaded successfully!")
    st.dataframe(company_df.head())

    if st.button("Train Model on Company Data", key="train_button"):
        try:
            df = company_df.copy()
            df = df.dropna()
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
            df["Education Level"] = df["Education Level"].map({"Bachelor's": 0, "Master's": 1, "PhD": 2})
            job_map = {
                "Software Engineer": 0, "Data Analyst": 1, "Senior Manager": 2,
                "Sales Associate": 3, "Director": 4, "Marketing Analyst": 5
            }
            df["Job Title"] = df["Job Title"].map(job_map)

            X = df.drop("Salary", axis=1)
            y = df["Salary"]

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model = XGBRegressor()
            model.fit(x_train, y_train)

            y_pred = model.predict(x_test)
            r2 = r2_score(y_test, y_pred)
            from math import sqrt
            rmse = sqrt(mean_squared_error(y_test, y_pred))
            acc = 100 - mean_absolute_percentage_error(y_test, y_pred) * 100

            model_scores['Company Model'] = {'R²': r2, 'RMSE': rmse, 'Accuracy': acc}

            joblib.dump(model, "company_model.pkl")
            st.success(f"✅ Custom model trained! Accuracy: {acc:.2f}%")
            use_custom = True
            custom_model = model

            # --- Feature Importance Display ---
            st.subheader("📌 Feature Importance")
            importance = model.feature_importances_
            features = ["Age", "Gender", "Education", "Job Title", "Experience"]
            fig, ax = plt.subplots()
            ax.barh(features, importance)
            ax.set_xlabel("Importance")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during training: {e}")

    # --- Batch Prediction and Download ---
    st.subheader("📅 Batch Predict & Download")
    if st.button("Download Predictions"):
        try:
            df = company_df.copy()
            df = df.dropna()
            df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})
            df["Education Level"] = df["Education Level"].map({"Bachelor's": 0, "Master's": 1, "PhD": 2})
            job_map = {
                "Software Engineer": 0, "Data Analyst": 1, "Senior Manager": 2,
                "Sales Associate": 3, "Director": 4, "Marketing Analyst": 5
            }
            df["Job Title"] = df["Job Title"].map(job_map)

            X_all = df.drop("Salary", axis=1)
            model = joblib.load("company_model.pkl")
            df["Predicted Salary"] = model.predict(X_all)
            csv = df.to_csv(index=False)
            st.download_button("📅 Download CSV", csv, "predicted_salaries.csv", "text/csv")
        except Exception as e:
            st.error(f"Failed to generate predictions: {e}")

# --- Prediction Form ---
st.subheader("🔍 Predict Salary")

age = st.number_input("Age", 18, 65, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
job_title = st.selectbox("Job Title", ["Software Engineer", "Data Analyst", "Senior Manager", "Director", "Sales Associate", "Marketing Analyst"])
experience = st.number_input("Years of Experience", 0, 40, 5)

model_choice = st.radio("Use which model?", ["General Model", "Company Model (if trained)"])

if st.button("Predict Salary", key="predict_button"):
    gender_val = 0 if gender == "Female" else 1
    edu_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
    job_map = {
        "Software Engineer": 0, "Data Analyst": 1, "Senior Manager": 2,
        "Sales Associate": 3, "Director": 4, "Marketing Analyst": 5
    }

    input_data = np.array([[age, gender_val, edu_map[education], job_map[job_title], experience]])

    try:
        if model_choice == "Company Model (if trained)":
            model = joblib.load("company_model.pkl")
        else:
            model = general_model

        prediction = model.predict(input_data)[0]
        st.success(f"💰 Predicted Salary: ₹{int(prediction):,}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# --- Optional: Model Comparison (if available) ---
if model_scores:
    st.subheader("📊 Model Performance Comparison")
    scores_df = pd.DataFrame(model_scores).T
    st.bar_chart(scores_df[["R²", "RMSE"]])



