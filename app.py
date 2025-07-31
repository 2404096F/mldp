import streamlit as st
import pandas as pd
import joblib

st.markdown("""
<style>
    .stApp {
        background-image: linear-gradient(to top, rgba(0,0,0,0.8), rgba(0,0,0,0.4)),
        url("https://omy.sg/wp-content/uploads/2023/12/What-Documents-Are-Required-for-A-Personal-Loan-In-Singapores.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
</style>
""", unsafe_allow_html=True)

model = joblib.load('model.pkl')

columns = [
    "loan_int_rate",
    "person_income",
    "loan_percent_income",
    "loan_amnt",
    "person_age",
    "person_emp_exp",
    "credit_score",
    "cb_person_cred_hist_length",
    "person_education_Bachelor",
    "person_education_Doctorate",
    "person_education_High School",
    "person_education_Master",
    "person_home_ownership_MORTGAGE",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_DEBTCONSOLIDATION",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "person_education_Associate"
]

st.title("Bank Loan Approval Prediction")
st.markdown("Fill in your details below to check your loan approval status.")

education_options = [
    "Bachelor", "Doctorate", "High School", "Master", "Associate"  # All options in your model columns
]
home_ownership_options = [
    "MORTGAGE", "OTHER", "OWN", "RENT"
]
loan_intent_options = [
    "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL",
    "PERSONAL", "VENTURE"
]
previous_defaults_options = ["No", "Yes"]

features_dict = dict.fromkeys(columns, 0)

# Numeric input fields
age = st.number_input(
    "Age", min_value=18, value=30, max_value=100,
    help="Your current age. Most banks require borrowers to be adults."
)
income = st.number_input(
    "Annual Income ($)", min_value=0,
    help="Your yearly total income helps determine loan affordability."
)
emp_exp = st.number_input(
    "Years of Employment Experience", min_value=0, max_value=50,
    help="Years you've been employed."
)
loan_amnt = st.number_input(
    "Loan Amount ($)", min_value=0,
    help="Total loan amount you wish to borrow."
)
loan_int_rate = st.number_input(
    "Max Willing Interest Rate (%)", min_value=0.0, max_value=99.0, value=10.0,
    help="Highest interest rate you'd accept."
)
cred_hist_length = st.number_input(
    "Credit History Length (years)", min_value=0, max_value=50,
    help="How long you’ve had a credit history."
)
credit_score = st.number_input(
    "Credit Score", min_value=0, max_value=900, value=650,
    help="A higher score increases your chances of approval."
)

# Categorical selects
education = st.selectbox(
    "Education", education_options,
    help="Highest degree obtained."
)
home_ownership = st.selectbox(
    "Home Ownership", home_ownership_options,
    help="Your home ownership status."
)
loan_intent = st.selectbox(
    "Loan Purpose", loan_intent_options,
    help="Main reason for your loan application."
)

# One-hot encoding population
features_dict[f'person_education_{education}'] = 1
features_dict[f'person_home_ownership_{home_ownership}'] = 1
features_dict[f'loan_intent_{loan_intent}'] = 1
# Derived feature
loan_percent_income = loan_amnt / income if income > 0 else 0

features_dict["person_age"] = age
features_dict["person_income"] = income
features_dict["person_emp_exp"] = emp_exp
features_dict["loan_amnt"] = loan_amnt
features_dict["loan_int_rate"] = loan_int_rate
features_dict["loan_percent_income"] = loan_percent_income
features_dict["cb_person_cred_hist_length"] = cred_hist_length
features_dict["credit_score"] = credit_score

# Recommendations
recommendations = []
if income < 30000:
    recommendations.append("**Increase your annual income** to improve your approval odds.")
if credit_score < 650:
    recommendations.append("**Raise your credit score** by paying bills on time and reducing debt.")
if features_dict['previous_loan_defaults_on_file_No']:
    recommendations.append("**Avoid future loan defaults**, as history of defaults severely impacts approval.")
if loan_percent_income > 0.4:
    recommendations.append("**Lower the loan amount** or increase income so the loan is a smaller share of your income.")

if recommendations:
    st.info("### Tips to Improve Approval Odds\n" + "\n".join(recommendations))
else:
    st.success("You have strong approval odds.")

# Compile the array for the model, always include every column the model expects
input_df = pd.DataFrame([features_dict])[columns]
features = input_df.values

if st.button("Predict Loan Approval"):
    pred = model.predict(features)
    prob = model.predict_proba(features)[0][1]
    if pred[0] == 1:
        st.success(f"✅ Loan likely approved! (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ Loan likely NOT approved. (Confidence: {1-prob:.2f})")
