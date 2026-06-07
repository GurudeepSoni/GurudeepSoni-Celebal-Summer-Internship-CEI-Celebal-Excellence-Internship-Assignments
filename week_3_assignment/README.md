# 🌍 Country Intelligence System

## Project Overview

This project was developed as part of the Celebal Technologies Internship Program.

The objective of the project is to predict the development status of a country and group similar countries using machine learning techniques.

The project includes data preprocessing, exploratory data analysis (EDA), classification, clustering, model saving, and deployment using Streamlit.

---

## Dataset

The dataset contains socio-economic and health indicators of different countries.

### Features Used

* Child Mortality
* Exports
* Health
* Imports
* Income
* Inflation
* Life Expectancy
* Total Fertility
* GDP Per Capita

---

## Machine Learning Techniques

### Classification

Random Forest Classifier was used to predict the development status of a country.

### Clustering

K-Means Clustering was used to group countries with similar characteristics.

---

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Scaling
5. Classification Model Training
6. Clustering Model Training
7. Model Evaluation
8. Model Saving using Joblib
9. Streamlit Application Development
10. Deployment

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

---

## Project Structure

week_3_assignment/

├── app.py

├── Country-data.csv

├── country_classifier.pkl

├── country_segmenter.pkl

├── scaler.pkl

├── requirements.txt

└── week3.ipynb

---

## Run Locally

Create Virtual Environment:

python -m venv venv

Activate Environment:

Windows:

venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Run Streamlit App:

streamlit run app.py

---

## Output

The application allows users to:

* Enter country indicators
* Predict development status
* View cluster assignment
* Store prediction history
* Download prediction history

---

## Developed By

Gurudeep Soni

Celebal Technologies Internship Project
