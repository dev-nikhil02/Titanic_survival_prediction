import streamlit as st
import numpy as np
import pandas as pd
import joblib

import os
import streamlit as st

st.write("Current working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())


# Load model and scaler
model = joblib.load("random_forest_model_titanic.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure it's the same one used during training

# Streamlit page config
st.set_page_config(page_title="🚢 Titanic Survival Predictor", layout="centered")
st.title("🚢 Titanic Survival Prediction App")

# Title mapping

title_mapping = {
    0: "Mr", 1: "Miss", 2: "Mrs", 3: "Master", 4: "Dr", 5: "Rev",
    6: "Col", 7: "Major", 8: "Mlle", 9: "Countess", 10: "Ms",
    11: "Lady", 12: "Don", 13: "Mme", 14: "Capt", 15: "Sir", 16: "Jonkheer"
}

# Create reverse mapping for display and selection
title_options = list(title_mapping.items())  # [(0, 'Mr'), (1, 'Miss'), ...]


# ---- USER INPUTS ----
st.header("🧍 Enter Passenger Information")
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Female", "Male"])
age = st.slider("Age (in years)", 0.0, 100.0, 30.0)
fare = st.slider("Fare Paid (in ₹ or $)", 0.0, 600.0, 50.0)

sibsp = st.slider("Number of Siblings/Spouses Aboard (SibSp)", 0, 8, 0)
parch = st.slider("Number of Parents/Children Aboard (Parch)", 0, 6, 0)

embarked = st.selectbox("Port of Embarkation", ["Cherbourg", "Queenstown", "Southampton"])
selected_title = st.selectbox(
    "Select Title",
    options=title_options,
    format_func=lambda x: x[1]  # Show title name
)
# Extract encoded value for model input
title_encoded = selected_title[0]


# ---- FEATURE ENGINEERING ----
sex = 1 if sex == "Male" else 0
embarked_map = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2}
embarked = embarked_map[embarked]

familysize = sibsp + parch +1
is_alone = 1 if familysize == 1 else 0

# ---- SCALING ----
scaled_df = pd.DataFrame([[age, fare, familysize]], columns=["Age", "Fare", "FamilySize"])
scaled_values = scaler.transform(scaled_df)
scaled_age, scaled_fare, scaled_familysize = scaled_values[0]

# ---- FINAL INPUT TO MODEL ----
model_input = np.array([[pclass, sex, scaled_age, scaled_fare, embarked, title_encoded, scaled_familysize, is_alone]])
columns = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
input_df = pd.DataFrame(model_input, columns=columns)

# ---- PREDICTION ----
if st.button("🔮 Predict Survival"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][prediction]

    st.subheader("🎯 Prediction Result")
    if prediction == 1:
        st.success(f"✅ Congratulation your survival chances are {probability*100:.2f}%.")
    else:
        st.error(f"❌ The passenger **would NOT have survived** with {probability*100:.2f}% confidence.")

# ---- FOOTER ----
st.markdown("---")
st.markdown("Project by - **Nikhil Agarwal**.")
