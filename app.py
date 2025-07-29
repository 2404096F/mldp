import streamlit as st
import pandas as pd
import joblib
import base64

def set_blurry_bg(image_file, blur_px=8):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        filter: blur({blur_px}px);
        /* Make child containers unblurred */
    }}
    /* Optional: remove blur for Streamlit widgets and main area */
    .main, .block-container {{
        backdrop-filter: blur(0px)!important;
        background: rgba(255,255,255,0.80); /* semi-transparent to view blurry bg */
        border-radius: 8px;
        padding: 1rem;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call this early in your app
set_blurry_bg("image.jpg", blur_px=8)

# ---- Load trained model ----
model = joblib.load('model.pkl')

# ---- List all feature columns in correct order ----
columns = [
    'person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate', 'loan_percent_income',
    'cb_person_cred_hist_length', 'credit_score',
    'person_education_Bachelor', 'person_education_Doctorate', 'person_education_High School', 'person_education_Master',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
    'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL', 'loan_intent_PERSONAL',
    'loan_intent_VENTURE',
    'previous_loan_defaults_on_file_Yes',
    'income_band_Medium', 'income_band_High', 'income_band_Very High',
    'credit_score_band_Fair', 'credit_score_band_Good', 'credit_score_band_Excellent'
]

st.title("Bank Loan Approval Prediction")
st.markdown("Fill in your details below to check your loan approval status.")

# Raw numeric input fields
age = st.number_input("Age", min_value=18, value=30, max_value=100)
income = st.number_input("Annual Income ($)", min_value=0)
emp_exp = st.number_input("Years of Employment Experience", min_value=0, max_value=50)
loan_amnt = st.number_input("Loan Amount ($)", min_value=0)
loan_int_rate = st.number_input("Max Willing Interest Rate (%)", min_value=0.0, max_value=99.0, value=10.0)
cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, max_value=50)
credit_score = st.number_input("Credit Score", min_value=0, max_value=900, value=650)

# Categorical selects
education = st.selectbox("Education", ["Bachelor", "Doctorate", "High School", "Master"])
home_ownership = st.selectbox("Home Ownership", ["OTHER", "OWN", "RENT"])
loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE"])
previous_defaults = st.selectbox("Did you fail to repay any previous loans?", ["No", "Yes"])

# Derived feature
if income > 0:
    loan_percent_income = loan_amnt / income
else:
    loan_percent_income = 0

# --- Calculate income bands ---
income_band_Medium = income_band_High = income_band_Very_High = 0
if income > 0: # Prevent division by zero
    # Replace thresholds with those from your training
    if income < 40000:
        income_band_Medium = 1
    elif income < 70000:
        income_band_High = 1
    else:
        income_band_Very_High = 1

# --- Calculate credit score bands ---
credit_score_band_Fair = credit_score_band_Good = credit_score_band_Excellent = 0
if credit_score < 650:
    credit_score_band_Fair = 1
elif credit_score < 750:
    credit_score_band_Good = 1
else:
    credit_score_band_Excellent = 1

# --- Build one-hot and feature vector in order ---
features_dict = dict.fromkeys(columns, 0)
features_dict['person_age'] = age
features_dict['person_income'] = income
features_dict['person_emp_exp'] = emp_exp
features_dict['loan_amnt'] = loan_amnt
features_dict['loan_int_rate'] = loan_int_rate
features_dict['loan_percent_income'] = loan_percent_income
features_dict['cb_person_cred_hist_length'] = cred_hist_length
features_dict['credit_score'] = credit_score
features_dict[f'person_education_{education}'] = 1
features_dict[f'person_home_ownership_{home_ownership}'] = 1
features_dict[f'loan_intent_{loan_intent}'] = 1
features_dict['previous_loan_defaults_on_file_Yes'] = int(previous_defaults == "Yes")
features_dict['income_band_Medium'] = income_band_Medium
features_dict['income_band_High'] = income_band_High
features_dict['income_band_Very High'] = income_band_Very_High
features_dict['credit_score_band_Fair'] = credit_score_band_Fair
features_dict['credit_score_band_Good'] = credit_score_band_Good
features_dict['credit_score_band_Excellent'] = credit_score_band_Excellent

# Compile final feature array
input_df = pd.DataFrame([features_dict])[columns]
features = input_df.values

st.markdown("---")

# ---- Prediction ----
if st.button("Predict Loan Approval"):
    pred = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    if pred[0] == 1:
        st.success(f"✅ Loan likely approved! (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Loan likely NOT approved. (Confidence: {1-prob:.2f})")

st.markdown("---")
st.caption("Results are based on a machine learning model trained on historical data. Contact your bank for formal applications.")
