import streamlit as st
import pickle
import numpy as np
import os



# Check if the pickle file exists
if not os.path.exists('oral_cancer_trained_models.pkl'):
    st.error("The pickle file 'oral_cancer_trained_models.pkl' was not found.")
    st.stop()

# Load the trained models from the pickle file
@st.cache_resource
def load_models():
    try:
        with open('oral_cancer_trained_models.pkl', 'rb') as f:
            trained_models = pickle.load(f)
        return trained_models
    except FileNotFoundError:
        st.error("Error: The pickle file 'oral_cancer_trained_models.pkl' was not found.")
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return None

trained_models = load_models()

if trained_models is None:
    st.error("Failed to load models. Please check the pickle file and dependencies.")
    st.stop()

# Streamlit UI
st.title("Oral Cancer Prediction")
st.write("Enter the patient's details to predict the likelihood of oral cancer.")

# Input fields for user data
st.sidebar.header("Patient Details")
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tobacco_use = st.sidebar.selectbox("Tobacco Use", ["No", "Yes"])
alcohol_consumption = st.sidebar.selectbox("Alcohol Consumption", ["No", "Yes"])
hpv_infection = st.sidebar.selectbox("HPV Infection", ["No", "Yes"])
betel_quid_use = st.sidebar.selectbox("Betel Quid Use", ["No", "Yes"])
chronic_sun_exposure = st.sidebar.selectbox("Chronic Sun Exposure", ["No", "Yes"])
poor_oral_hygiene = st.sidebar.selectbox("Poor Oral Hygiene", ["No", "Yes"])
diet = st.sidebar.selectbox("Diet (Fruits & Vegetables Intake)", ["Low", "Moderate", "High"])
family_history = st.sidebar.selectbox("Family History of Cancer", ["No", "Yes"])
compromised_immune_system = st.sidebar.selectbox("Compromised Immune System", ["No", "Yes"])
oral_lesions = st.sidebar.selectbox("Oral Lesions", ["No", "Yes"])
unexplained_bleeding = st.sidebar.selectbox("Unexplained Bleeding", ["No", "Yes"])
difficulty_swallowing = st.sidebar.selectbox("Difficulty Swallowing", ["No", "Yes"])
white_red_patches = st.sidebar.selectbox("White or Red Patches in Mouth", ["No", "Yes"])

# Convert categorical inputs to numerical values
gender = 0 if gender == "Male" else 1
tobacco_use = 0 if tobacco_use == "No" else 1
alcohol_consumption = 0 if alcohol_consumption == "No" else 1
hpv_infection = 0 if hpv_infection == "No" else 1
betel_quid_use = 0 if betel_quid_use == "No" else 1
chronic_sun_exposure = 0 if chronic_sun_exposure == "No" else 1
poor_oral_hygiene = 0 if poor_oral_hygiene == "No" else 1
diet = 0 if diet == "Low" else 1 if diet == "Moderate" else 2
family_history = 0 if family_history == "No" else 1
compromised_immune_system = 0 if compromised_immune_system == "No" else 1
oral_lesions = 0 if oral_lesions == "No" else 1
unexplained_bleeding = 0 if unexplained_bleeding == "No" else 1
difficulty_swallowing = 0 if difficulty_swallowing == "No" else 1
white_red_patches = 0 if white_red_patches == "No" else 1

# Create a feature vector for prediction
input_data = np.array([age, gender, tobacco_use, alcohol_consumption, hpv_infection, betel_quid_use,
                      chronic_sun_exposure, poor_oral_hygiene, diet, family_history,
                      compromised_immune_system, oral_lesions, unexplained_bleeding,
                      difficulty_swallowing, white_red_patches]).reshape(1, -1)

# Model selection
model_name = st.sidebar.selectbox("Select Model", list(trained_models.keys()))
model = trained_models[model_name]

# Predict button
if st.sidebar.button("Predict"):
    try:
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("The model predicts **Positive** for oral cancer.")
        else:
            st.success("The model predicts **Negative** for oral cancer.")

        pred_positive = prediction_proba[0][1]
        pred_negative = prediction_proba[0][0]
        st.write(f"Probability of Positive Class: {pred_positive * 100:.2f}%")
        st.write(f"Probability of Negative Class: {pred_negative * 100:.2f}%")
    except Exception as e:
        st.error(f"Error making prediction: {e}")