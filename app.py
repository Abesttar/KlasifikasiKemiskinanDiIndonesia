import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("kemiskinan_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load nama fitur
feature_names = pd.read_csv("feature_columns.txt", header=None).squeeze().tolist()

# UI Streamlit
st.set_page_config(page_title="Klasifikasi Kemiskinan", layout="centered")
st.title("🧮 Aplikasi Klasifikasi Tingkat Kemiskinan")

st.markdown("Masukkan data indikator sosial ekonomi melalui sidebar, lalu sistem akan memprediksi tingkat kemiskinan.")

# Sidebar input
st.sidebar.header("📝 Input Data")
user_input = []

# Konfigurasi batas slider default (opsional manual override)
slider_bounds = {
    "Pengeluaran per Kapita Disesuaikan (Ribu Rupiah/Orang/Tahun)": (5000, 20000),
    "Umur Harapan Hidup (Tahun)": (50, 80),
    "Rata-rata Lama Sekolah Penduduk 15+ (Tahun)": (5, 15),
    "Persentase rumah tangga yang memiliki akses terhadap sanitasi layak": (0, 100),
    "Persentase rumah tangga yang memiliki akses terhadap air minum layak": (0, 100),
    "Tingkat Pengangguran Terbuka": (0, 30),
    "Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)": (0, 40)
}

# Buat slider untuk setiap fitur
for feature in feature_names:
    min_val, max_val = slider_bounds.get(feature, (0.0, 100.0))
    value = st.sidebar.slider(
        feature,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float((min_val + max_val) / 2),
        step=0.1
    )
    user_input.append(value)

# Prediksi
if st.sidebar.button("🔍 Prediksi"):
    input_array = np.array([user_input])
    prediction = model.predict(input_array)[0]
    st.success(f"📊 Hasil prediksi tingkat kemiskinan: **{prediction}**")
