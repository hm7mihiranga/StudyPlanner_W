import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")  # Make sure you saved this after training

# Mappings
Parental_Involvement_mapping = {'Low':1,'Medium':2,'High':3}
Access_to_Resources_mapping = {'Low':1,'Medium':2,'High':3}
Extracurricular_Activities_mapping = {'No':0, 'Yes':1}
Motivation_Level_mapping = {'Low':1,'Medium':2,'High':3}
Internet_Access_mapping = {'No':0, 'Yes':1}
Family_Income_mapping = {'Low':1,'Medium':2,'High':3}
Teacher_Quality_mapping = {'Unknown':0,'Low':1,'Medium':2,'High':3}
School_Type_mapping = {'Public':1,'Private':2}
Peer_Influence_mapping = {'Positive':1,'Neutral':0,'Negative':-1}
Learning_Disabilities_mapping = {'No':0, 'Yes':1}
Parental_Education_Level_mapping = {'Unknown':0,'High School':1,'College':2,'Postgraduate':3}
Distance_from_Home_mapping = {'Unknown':0,'Near':1,'Moderate':2,'Far':3}
Gender_mapping = {"Male":0,"Female":1}

# UI Setup
st.set_page_config(page_title="📘 Exam Score Predictor", layout="centered")
st.title("🎓 Student Exam Score Predictor")
st.markdown("Fill in the student details below to predict their expected **Exam Score**.")

# Inputs
hours_studied = st.slider("🕒 Hours Studied (per week)", 0, 50, 20)
attendance = st.slider("📈 Attendance (%)", 0, 100, 80)
parental_involvement = st.selectbox("👪 Parental Involvement", list(Parental_Involvement_mapping.keys()))
access_to_resources = st.selectbox("📚 Access to Resources", list(Access_to_Resources_mapping.keys()))
extracurricular = st.selectbox("⚽ Extracurricular Activities", list(Extracurricular_Activities_mapping.keys()))
sleep_hours = st.slider("😴 Sleep Hours (per night)", 0, 12, 7)
previous_scores = st.slider("📊 Previous Exam Score", 0, 100, 70)
motivation = st.selectbox("🔥 Motivation Level", list(Motivation_Level_mapping.keys()))
internet_access = st.selectbox("🌐 Internet Access", list(Internet_Access_mapping.keys()))
tutoring_sessions = st.slider("👨‍🏫 Tutoring Sessions (per month)", 0, 20, 2)
family_income = st.selectbox("💰 Family Income", list(Family_Income_mapping.keys()))
teacher_quality = st.selectbox("📏 Teacher Quality", list(Teacher_Quality_mapping.keys()))
school_type = st.selectbox("🏫 School Type", list(School_Type_mapping.keys()))
peer_influence = st.selectbox("🧍 Peer Influence", list(Peer_Influence_mapping.keys()))
physical_activity = st.slider("🏃 Physical Activity (hours/week)", 0, 15, 4)
learning_disability = st.selectbox("🧠 Learning Disability", list(Learning_Disabilities_mapping.keys()))
parental_education = st.selectbox("🎓 Parental Education", list(Parental_Education_Level_mapping.keys()))
distance_home = st.selectbox("📍 Distance from Home", list(Distance_from_Home_mapping.keys()))
gender = st.selectbox("🧑 Gender", list(Gender_mapping.keys()))

# Transform input
input_features = [
    np.sqrt(hours_studied),  # square root transformed
    attendance,
    Parental_Involvement_mapping[parental_involvement],
    Access_to_Resources_mapping[access_to_resources],
    Extracurricular_Activities_mapping[extracurricular],
    sleep_hours,
    previous_scores,
    Motivation_Level_mapping[motivation],
    Internet_Access_mapping[internet_access],
    tutoring_sessions,
    Family_Income_mapping[family_income],
    Teacher_Quality_mapping[teacher_quality],
    School_Type_mapping[school_type],
    Peer_Influence_mapping[peer_influence],
    physical_activity,
    Learning_Disabilities_mapping[learning_disability],
    Parental_Education_Level_mapping[parental_education],
    Distance_from_Home_mapping[distance_home],
    Gender_mapping[gender]
]

# Predict
if st.button("🔍 Predict Exam Score"):
    input_array = np.array(input_features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)  # Scale input as done during training
    pred_sqrt = model.predict(input_scaled)[0]
    final_score = np.round(pred_sqrt ** 2, 2)  # Reverse square root transformation

    if final_score > 100:
        final_score = 100.0  # Cap to realistic max if needed

    st.success(f"🎯 Predicted Exam Score: **{final_score} / 100**")
