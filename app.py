import streamlit as st
import numpy as np
import joblib

# Load model and scaler

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title

st.title("Medical Insurance Cost Prediction")

st.write("KNN Regression Project")

# Inputs

age = st.number_input("Age", 1, 100, 25)

sex = st.selectbox("Gender", ["Male", "Female"])

bmi = st.number_input("BMI", 10.0, 60.0, 25.0)

children = st.number_input("Children", 0, 10, 0)

smoker = st.selectbox("Smoker", ["Yes", "No"])

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

# Encoding

sex = 1 if sex == "Male" else 0

smoker = 1 if smoker == "Yes" else 0

region_dict = {
    "northeast": 0,
    "northwest": 1,
    "southeast": 2,
    "southwest": 3
}

region = region_dict[region]

# Prediction Button

if st.button("Predict"):

    data = np.array([
        [age, sex, bmi, children, smoker, region]
    ])

    data = scaler.transform(data)

    prediction = model.predict(data)

    st.success(
        f"Predicted Insurance Cost: ${prediction[0]:.2f}"
    )
