import streamlit as st
import pandas as pd
import pickle

# --- LOAD MODEL ---
with open('model_xgb.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Diabetes Prediction", layout="centered")
st.title("🩺 Diabetes Prediction App")
st.write("Masukkan data pasien di bawah ini untuk memprediksi risiko diabetes.")

# --- INPUT UI ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=1, max_value=120, value=30)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, value=25.0)
    hba1c = st.number_input("HbA1c Level", min_value=3.0, value=5.5)
    blood_glucose = st.number_input("Blood Glucose Level", min_value=50, value=100)
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])

with col2:
    race = st.selectbox("Ras", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"])
    smoking = st.selectbox("Riwayat Merokok", ["No Info", "current", "ever", "former", "never", "not current"])
    hypertension = st.selectbox("Hipertensi", [0, 1], help="0: Tidak, 1: Ya")
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1], help="0: Tidak, 1: Ya")

# --- PREPROCESSING ---
# Urutan kolom harus SAMA PERSIS dengan X_train pada saat training:
# age, race:AfricanAmerican, race:Asian, race:Caucasian, race:Hispanic, race:Other,
# hypertension, heart_disease, bmi, hbA1c_level, blood_glucose_level,
# gender_Male, smoking_history_current, smoking_history_ever, smoking_history_former,
# smoking_history_never, smoking_history_not current
def preprocess_input():
    data = {
        'age': age,

        # Race (kolom biner asli dari dataset — tidak di-encode, langsung dipakai)
        'race:AfricanAmerican': 1 if race == "AfricanAmerican" else 0,
        'race:Asian':           1 if race == "Asian" else 0,
        'race:Caucasian':       1 if race == "Caucasian" else 0,
        'race:Hispanic':        1 if race == "Hispanic" else 0,
        'race:Other':           1 if race == "Other" else 0,

        'hypertension':         hypertension,
        'heart_disease':        heart_disease,
        'bmi':                  bmi,
        'hbA1c_level':          hba1c,   # nama kolom asli: hbA1c_level (h kecil)
        'blood_glucose_level':  blood_glucose,

        # Gender: drop_first=True → "Female" dibuang, hanya tersisa gender_Male
        'gender_Male': 1 if gender == "Male" else 0,

        # Smoking: drop_first=True → "No Info" dibuang sebagai baseline
        'smoking_history_current':     1 if smoking == "current" else 0,
        'smoking_history_ever':        1 if smoking == "ever" else 0,
        'smoking_history_former':      1 if smoking == "former" else 0,
        'smoking_history_never':       1 if smoking == "never" else 0,
        'smoking_history_not current': 1 if smoking == "not current" else 0,
    }
    return pd.DataFrame([data])

input_df = preprocess_input()

# --- PREDIKSI ---
st.divider()
if st.button("Cek Hasil Prediksi"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error(f"Hasil: **Positif Diabetes** (Probabilitas: {probability:.2%})")
    else:
        st.success(f"Hasil: **Negatif Diabetes** (Probabilitas Diabetes: {probability:.2%})")