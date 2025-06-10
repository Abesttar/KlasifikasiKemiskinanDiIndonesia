import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("kemiskinan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load nama fitur
feature_names = pd.read_csv("feature_columns.txt", header=None).squeeze().tolist()

# Konfigurasi halaman
st.set_page_config(page_title="Klasifikasi Kemiskinan", layout="centered")
st.title("🧮 Klasifikasi Tingkat Kemiskinan")

st.markdown("Masukkan data indikator sosial ekonomi melalui sidebar, lalu sistem akan memprediksi tingkat kemiskinan.")

# Sidebar input
st.sidebar.header("📝 Input Data")
user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, format="%.2f")
    user_input.append(value)

# Prediksi
if st.sidebar.button("🔍 Prediksi"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]
    st.success(f"📊 Hasil prediksi tingkat kemiskinan: **{prediction}**")
