import pickle
import streamlit as st
import numpy as np

# membaca model
try:
    Jantung_model = pickle.load(open('Jantung_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("File model tidak ditemukan. Pastikan path file sudah benar.")
    st.stop()
# judul web
st.title('Data Mining Prediksi Jantung')

# membagi kolom
col1, col2 = st.columns(2)

with col1 :
    age = st.text_input ('Input age')

with col2 :
    anaemia = st.text_input ('Input Nilai anaemia')

with col1 :
    creatinine_phosphokinase = st.text_input ('Input Nilai creatinine_phosphokinase')

with col2 :
    diabetes = st.text_input ('Input Nilai diabetes')

with col1 :
    ejection_fraction = st.text_input ('Input Nilai ejection_fraction')

with col2 :
    high_blood_pressure = st.text_input ('Input Nilai high_blood_pressure')

with col1 :
    platelets = st.text_input ('Input Nilai platelets')

with col2 :
    serum_creatinine = st.text_input ('Input Nilai serum_creatinine')

with col1 :
    serum_sodium = st.text_input ('Input Nilai serum_sodium')

with col2 :
    sex = st.text_input ('Input Nilai sex')

with col1 :
    smoking = st.text_input ('Input Nilai smoking')

with col2 :
    time = st.text_input ('Input Nilai time')
    

# code untuk prediksi
jantung_diagnosis = ''

# membuat tombol untuk prediksi
if st.button('Test Prediksi Jantung'):
    try:
        # Konversi input menjadi float dan numerik
        input_data = [
            float(age), 
            int(anaemia.split(':')[0]), 
            float(creatinine_phosphokinase), 
            int(diabetes.split(':')[0]), 
            float(ejection_fraction), 
            int(high_blood_pressure.split(':')[0]),
            float(platelets), 
            float(serum_creatinine), 
            float(serum_sodium), 
            int(sex.split(':')[0]), 
            int(smoking.split(':')[0]), 
            float(time)
        ]

        # Konversi ke array 2D
        input_data = np.array(input_data).reshape(1, -1)

        # Normalisasi data menggunakan scaler
        input_data_scaled = scaler.transform(input_data)

        # Prediksi menggunakan model
        jantung_prediction = jantung_model.predict(input_data_scaled)

        # Menentukan hasil prediksi
        if jantung_prediction[0] == 1:
            jantung_diagnosis = 'Pasien tidak terkena penyakit jantung.'
        else:
            jantung_diagnosis = 'Pasien terkena penyakit jantung.'

        # Tampilkan hasil
        st.success(jantung_diagnosis)

    except ValueError as e:
        st.error(f"Input tidak valid: {e}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
       
