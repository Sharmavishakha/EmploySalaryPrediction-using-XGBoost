# 💼 Salary Prediction Web App

A machine learning-based Streamlit web app that predicts salaries based on employee details.

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
