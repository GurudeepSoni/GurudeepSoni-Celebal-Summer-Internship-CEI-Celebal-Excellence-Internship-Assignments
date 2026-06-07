import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Country Intelligence System",
    page_icon="🌍",
    layout="wide"
)

# Load Models
classifier = joblib.load("country_classifier.pkl")
segmenter = joblib.load("country_segmenter.pkl")
scaler = joblib.load("scaler.pkl")

# Load Dataset
df = pd.read_csv("Country-data.csv")

# Store Prediction History
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Home",
        "Prediction",
        "Search History",
        "About"
    ]
)

# =========================
# Home Page
# =========================

if page == "Home":

    st.title("🌍 Country Intelligence System")

    st.write(
        """
        This project predicts the development status of a country
        and groups similar countries using machine learning models.

        The application uses Random Forest for classification
        and K-Means for clustering.
        """
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Countries", df.shape[0])

    with col2:
        st.metric("Features Used", 9)

    with col3:
        st.metric("Model Accuracy", "94.12%")

    st.markdown("---")

    st.subheader("Dataset Features")

    st.write("""
    • Child Mortality

    • Exports

    • Health

    • Imports

    • Income

    • Inflation

    • Life Expectancy

    • Total Fertility

    • GDP Per Capita
    """)

# =========================
# Prediction Page
# =========================

elif page == "Prediction":

    st.title("Country Prediction")

    st.write(
        "Enter the country indicators below and click Predict."
    )

    col1, col2 = st.columns(2)

    with col1:

        child_mort = st.number_input(
            "Child Mortality",
            value=20.0,
            help="Example: 20"
        )

        exports = st.number_input(
            "Exports",
            value=40.0,
            help="Example: 40"
        )

        health = st.number_input(
            "Health",
            value=6.0,
            help="Example: 6"
        )

        imports = st.number_input(
            "Imports",
            value=45.0,
            help="Example: 45"
        )

        income = st.number_input(
            "Income",
            value=12000.0,
            help="Example: 12000"
        )

    with col2:

        inflation = st.number_input(
            "Inflation",
            value=5.0,
            help="Example: 5"
        )

        life_expec = st.number_input(
            "Life Expectancy",
            value=70.0,
            help="Example: 70"
        )

        total_fer = st.number_input(
            "Total Fertility",
            value=2.0,
            help="Example: 2"
        )

        gdpp = st.number_input(
            "GDP Per Capita",
            value=8000.0,
            help="Example: 8000"
        )

    if st.button("Predict"):

        features = np.array([
            [
                child_mort,
                exports,
                health,
                imports,
                income,
                inflation,
                life_expec,
                total_fer,
                gdpp
            ]
        ])

        scaled_features = scaler.transform(features)

        prediction = classifier.predict(
            scaled_features
        )[0]

        cluster = segmenter.predict(
            scaled_features
        )[0]

        status = (
            "Developed Country"
            if prediction == 0
            else "Developing Country"
        )

        st.subheader("Prediction Result")

        st.success(status)

        st.info(f"Cluster Assigned: {cluster}")

        st.session_state.history.append({
            "Time": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "Prediction": status,
            "Cluster": cluster
        })

# =========================
# Search History
# =========================

elif page == "Search History":

    st.title("Search History")

    if len(st.session_state.history) == 0:

        st.warning(
            "No prediction history available."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

        csv = history_df.to_csv(index=False)

        st.download_button(
            label="Download History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

# =========================
# About Page
# =========================

elif page == "About":

    st.title("About Project")

    st.write("""
    This project was developed to classify and
    segment countries using socio-economic indicators.

    Project Workflow:

    • Data Cleaning

    • Exploratory Data Analysis (EDA)

    • Classification Models

    • Ensemble Learning

    • Clustering Techniques

    • Streamlit Deployment

    Machine Learning Models Used:

    • Random Forest Classifier

    • K-Means Clustering

    Tools Used:

    • Python

    • Pandas

    • NumPy

    • Scikit-Learn

    • Streamlit
    """)

# =========================
# Footer
# =========================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;">
        <b>Developed by Gurudeep Soni</b>
    </div>
    """,
    unsafe_allow_html=True
)
