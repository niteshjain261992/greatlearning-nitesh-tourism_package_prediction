import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "tourism_package_pridiction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the Tourism Package.
Enter customer details below to get a prediction.
""")

# Input fields for the features
age = st.number_input("Age", 18, 90, 35)
type_of_contact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", 1, 60, 10)
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
number_of_person_visiting = st.number_input("Number of People Visiting", 1, 10, 2)
preferred_property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
number_of_trips = st.number_input("Number of Trips Annually", 0, 20, 2)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
number_of_children_visiting = st.number_input("Number of Children Visiting", 0, 5, 0)
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
monthly_income = st.number_input("Monthly Income", 0, 100000, 30000)
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score (1-5)", [1, 2, 3, 4, 5])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
number_of_followups = st.number_input("Number of Follow-ups", 0, 10, 3)

# Create a DataFrame from the input data
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups
}])

if st.button("Predict Purchase"):
    # Predict probabilities and apply threshold
    prediction_proba = model.predict_proba(input_data)[:, 1]
    classification_threshold = 0.45 # Using the same threshold as in training
    prediction = (prediction_proba >= classification_threshold).astype(int)[0]

    result = "likely to purchase" if prediction == 1 else "not likely to purchase"
    st.subheader("Prediction Result:")
    st.success(f"The customer is **{result}** the Wellness Tourism Package.")
