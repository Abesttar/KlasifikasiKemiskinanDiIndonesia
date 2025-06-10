import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("kemiskinan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load nama fitur
feature_names = pd.read_csv("feature_columns.txt", header=None).squeeze().tolist()

# Load dataset untuk dapatkan min-max tiap fitur
# (jika tidak ada, kamu bisa buat sendiri manual)
try:
    df = pd.read_csv("Klasifikasi Tingkat Kemiskinan di Indonesia.csv", sep=';')
    df = df.select_dtypes(include=['float64', 'int64'])  # hanya numerik
except:
    df = None  # fallback kalau dataset tidak tersedia

# UI config
st.set_page_config(page_title="Klasifikasi Kemiskinan", layout="centered")
st.title("🧮 Aplikasi Klasifikasi Tingkat Kemiskinan")
st.markdown("Masukkan data indikator sosial ekonomi melalui sidebar, lalu sistem akan memprediksi tingkat kemiskinan.")

# Sidebar input
st.sidebar.header("📝 Input Data")
user_input = []

for feature in feature_names:
    if df is not None and feature in df.columns:
        min_val = float(df[feature].min())
        max_val = float(df[feature].max())
    else:
        min_val, max_val = 0.0, 100.0  # fallback range

    value = st.sidebar.slider(
        label=feature,
        min_value=round(min_val, 2),
        max_value=round(max_val, 2),
        value=round((min_val + max_val) / 2, 2),
        step=0.1
    )
    user_input.append(value)

# Prediksi
if st.sidebar.button("🔍 Prediksi"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]
    st.success(f"📊 Hasil prediksi tingkat kemiskinan: **{prediction}**")
