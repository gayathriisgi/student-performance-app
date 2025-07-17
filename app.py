import streamlit as st
import pandas as pd
import joblib

# Load model and training columns
model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Will the Student Pass Math?")

# Sidebar inputs
st.sidebar.header("📊 Enter Student Info:")

gender = st.sidebar.selectbox("Gender", ["male", "female"])
race = st.sidebar.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.sidebar.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.sidebar.selectbox("Lunch", ["standard", "free/reduced"])
prep = st.sidebar.selectbox("Test Preparation Course", ["none", "completed"])
reading = st.sidebar.slider("Reading Score", 0, 100, 50)
writing = st.sidebar.slider("Writing Score", 0, 100, 50)

# Build input DataFrame
input_dict = {
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parent_edu,
    'lunch': lunch,
    'test preparation course': prep,
    'reading score': reading,
    'writing score': writing
}

input_df = pd.DataFrame([input_dict])

# One-hot encode and align with training columns
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("🎯 Predict"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ The student is likely to PASS math. Confidence: {proba:.2%}")
    else:
        st.error(f"❌ The student is likely to FAIL math. Confidence: {1 - proba:.2%}")
