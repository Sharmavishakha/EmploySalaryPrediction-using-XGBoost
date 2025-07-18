# 💼 Salary Prediction Web App

A machine learning-based Streamlit web app that predicts salaries based on employee details.

**SalaryScope** is an interactive Machine Learning-based web application that predicts employee salaries based on key features such as age, gender, education level, job title, and years of experience.

Built using **Streamlit** and powered by **XGBoost**, this system provides both general salary predictions and company-specific training functionality, making it suitable for real-world HR and analytics use cases.

## 🔍 Features
- Predict salary using input fields
- Upload company-specific employee data to train a custom model
- Visualize feature importance
- Batch prediction + CSV download

## 📁 Required CSV Format
| Column | Description |
|--------|-------------|
| Age | Age (int) |
| Gender | Male or Female |
| Education Level | Bachelor's, Master's, or PhD |
| Job Title | Software Engineer, etc. |
| Years of Experience | int |
| Salary | int or float |

## 🛠️ Tech Stack
- Python, Streamlit, XGBoost, pandas, sklearn
- Built on Google Colab

## 🚀 Run Locally
```bash
pip install streamlit xgboost scikit-learn pandas joblib
streamlit run app.py
