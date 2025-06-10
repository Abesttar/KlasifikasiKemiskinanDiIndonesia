import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("kemiskinan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load nama kolom fitur
feature_names = pd.read_csv("feature_columns.txt", header=None).squeeze().tolist()

st.set_page_config(page_title="Klasifikasi Kemiskinan", layout="centered")
st.title("🧮 Aplikasi Klasifikasi Tingkat Kemiskinan")

st.markdown("Masukkan data indikator sosial ekonomi, lalu sistem akan memprediksi tingkat kemiskinan.")

# Ambil input dari user
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.2f")
    user_input.append(value)

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]
    st.success(f"Hasil prediksi tingkat kemiskinan: **{prediction}**")
