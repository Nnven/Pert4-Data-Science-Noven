import streamlit as st
import pandas as pd
import joblib

# 1. PERSIAPAN MUAT ARTEFAK DENGAN CACHE
@st.cache_resource
def load_artefak():
    scaler = joblib.load('scaler.joblib')
    model_kmeans = joblib.load('kmeans_model.joblib')
    return scaler, model_kmeans

scaler, model_kmeans = load_artefak()

# 2. UI STREAMLIT
st.title("🛒 Prediksi Segmen Pelanggan E-Commerce")
st.write("Aplikasi dasbor interaktif untuk memprediksi kelompok pelanggan berdasarkan perilaku belanja.")

st.markdown("---")

# Layout 2 kolom untuk input
col1, col2 = st.columns(2)

with col1:
    age_val = st.number_input("Age (Umur Pelanggan)", value=35)
    total_val = st.number_input("Total Amount (Pengeluaran $)", value=642.0)

with col2:
    session_val = st.number_input("Session Duration (Durasi Sesi Menit)", value=10.0)
    rating_val = st.slider("Customer Rating (Rating Kepuasan)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)

# Ringkasan Kategori Singkat
st.caption("💡 **Kategori:** Cluster 0 (Pelanggan Sangat Puas/Regular) | Cluster 1 (Pelanggan Kecewa) | Cluster 2 (Suka Browsing) | Cluster 3 (VIP / High-Spender)")

# 3. LOGIKA PREDIKSI
if st.button("Prediksi Cluster"):
    # Bungkus input menjadi DataFrame sesuai nama fitur
    input_data = pd.DataFrame(
        [[age_val, total_val, session_val, rating_val]], 
        columns=['Age', 'Total_Amount', 'Session_Duration_Minutes', 'Customer_Rating']
    )
    
    # Standarisasi data input menggunakan scaler
    scaled_input = scaler.transform(input_data)
    
    # Lakukan prediksi
    cluster_result = model_kmeans.predict(scaled_input)[0]
    
    # Keterangan label singkat per cluster
    labels = {
        0: "Pelanggan Sangat Puas / Regular 🌟",
        1: "Pelanggan Kecewa / Perlu Perhatian ⚠️",
        2: "Suka Browsing / Window Shopper 🔍",
        3: "Pelanggan VIP / High-Spender 💎"
    }
    
    # Tampilkan hasil
    st.success(f"Pelanggan ini masuk ke dalam: **Cluster {cluster_result} - {labels[cluster_result]}**")
    st.metric(label="Status Prediksi", value="Berhasil", delta="Selesai")