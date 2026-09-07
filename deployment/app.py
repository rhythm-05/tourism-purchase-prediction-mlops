
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# -----------------------------
# Load model from Hugging Face
# -----------------------------
MODEL_REPO = "rhythm0511/tourism-purchase-prediction"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="final_model.joblib",
    repo_type="model"
)

model = joblib.load(model_path)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="🏖️"
)

st.title("🏖️ Wellness Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the package.")

st.subheader("Customer Information")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
type_of_contact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Free Lancer", "Small Business", "Large Business"]
)
gender = st.selectbox("Gender", ["Male", "Female"])

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting", min_value=1, value=2
)

preferred_property_star = st.selectbox(
    "Preferred Property Star", [3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Divorced", "Married", "Unmarried"]
)

number_of_trips = st.number_input(
    "Number of Trips", min_value=0, value=2
)

passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])

number_of_children_visiting = st.number_input(
    "Number of Children Visiting", min_value=0, value=0
)

designation = st.selectbox(
    "Designation",
    ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income", min_value=0.0, value=20000.0
)

pitch_satisfaction_score = st.selectbox(
    "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
)

number_of_followups = st.number_input(
    "Number of Followups", min_value=0, value=3
)

duration_of_pitch = st.number_input(
    "Duration of Pitch", min_value=0, value=15
)

if st.button("Predict Purchase"):
    
    input_data = pd.DataFrame([{
        "Age": age,
        "TypeofContact": type_of_contact,
        "CityTier": city_tier,
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
        "NumberOfFollowups": number_of_followups,
        "DurationOfPitch": duration_of_pitch
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(
            f"Customer is likely to purchase the package "
            f"(Probability: {probability:.2%})"
        )
    else:
        st.warning(
            f"Customer is unlikely to purchase the package "
            f"(Probability: {probability:.2%})"
        )
