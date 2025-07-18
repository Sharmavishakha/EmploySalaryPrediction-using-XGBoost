# 💼 SalaryScope: A Streamlit-Powered AI System for Personalized Compensation Prediction

**SalaryScope** is an interactive Machine Learning-based web application that predicts employee salaries based on key features such as age, gender, education level, job title, and years of experience.

Built using **Streamlit** and powered by **XGBoost**, this system provides both general salary predictions and company-specific training functionality, making it suitable for real-world HR and analytics use cases.

---

## 🔍 Features

- 🎯 **Salary Prediction Form** – Enter employee attributes to get an instant salary estimate.
- 🏢 **Company-Specific Model Training** – Upload your own organizational data to train a customized model.
- 📊 **Feature Importance Visualization** – See which features impact salary most using interactive bar graphs.
- 📁 **Batch Prediction & Download** – Upload full datasets, predict salaries in bulk, and download results as CSV.
- 📈 **Model Metrics Display** – View performance metrics such as R² Score, RMSE, and accuracy.
- 🧠 **Adaptive Learning** – Automatically switches between the general model and custom-trained models.
- 📌 **Built-in CSV Format Guidance** – Interactive help section to avoid format errors during upload.

---

## 🧠 Problem Statement

HR professionals often struggle to determine fair and competitive salaries for potential and current employees. Factors like experience, education, and role demand play significant roles — but without data-driven insights, compensation can become biased or inconsistent.

This project solves that problem by providing a predictive system trained on reliable data and capable of being customized with real company data. It aims to bring transparency and intelligence to the salary estimation process.

---

## 📁 Required CSV Format for Custom Training

| Column               | Description                                 |
|----------------------|---------------------------------------------|
| `Age`                | Age of the employee (integer)               |
| `Gender`             | `Male` or `Female`                          |
| `Education Level`    | `Bachelor's`, `Master's`, or `PhD`          |
| `Job Title`          | e.g., `Software Engineer`, `Data Analyst`   |
| `Years of Experience`| Work experience in years (integer)          |
| `Salary`             | Current salary in INR (integer or float)    |

⚠️ **Note**: Ensure column names and values match the expected format exactly to avoid errors during training.

---

## 🛠️ Tech Stack

| Component     | Tools/Libraries Used                             |
|---------------|--------------------------------------------------|
| Frontend      | Streamlit                                        |
| Machine Learning | XGBoost, scikit-learn                         |
| Data Handling | pandas, numpy, matplotlib                        |
| Model Saving  | joblib                                           |
| Platform      | Google Colab, Local Execution                    |

---

## 📊 Model Performance

The XGBoost model achieved the highest accuracy among all tested algorithms:

| Model               | R² Score | RMSE (₹) | Accuracy |
|---------------------|----------|----------|----------|
| Linear Regression   | 0.89     | ₹15,782  | ~89.6%   |
| Random Forest       | 0.94     | ₹11,970  | ~93.5%   |
| Gradient Boosting   | 0.92     | ₹13,287  | ~92.0%   |
| **XGBoost (Final)** | **0.95** | **₹10,980** | **90.4%** |

---


## 🚀 Run Locally
```bash
pip install streamlit xgboost scikit-learn pandas joblib
streamlit run app.py
