import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load model dan data
@st.cache_resource
def load_recommender():
    with open('laptop_recommender.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load data dari file pickle
data = load_recommender()

# Cek tipe data dari pickle
if isinstance(data, dict):  # Jika data adalah dictionary
    df = data.get('data', pd.DataFrame())  # Ambil DataFrame jika ada
    similarity_matrix = data.get('similarity_matrix', None)  # Ambil matriks kemiripan
else:
    st.error("Format file pickle tidak sesuai. Harus berupa dictionary dengan key 'data' dan 'similarity_matrix'.")
    st.stop()

# Judul aplikasi
st.title("💻 Sistem Rekomendasi Laptop E-commerce")
st.markdown("""
Aplikasi ini menggunakan metode **Collaborative Filtering dengan Cosine Similarity** 
untuk merekomendasikan laptop berdasarkan produk yang dipilih. 
""")

# Dropdown untuk memilih produk
if 'name' in df.columns:
    product_name = st.selectbox('Pilih Laptop', df['name'])

    # Menampilkan rekomendasi produk
    if product_name and similarity_matrix is not None:
        st.subheader(f"Rekomendasi untuk '{product_name}'")
        
        # Cari index produk terpilih
        product_idx = df[df['name'] == product_name].index[0]
        
        # Ambil skor kemiripan produk
        product_scores = similarity_matrix[product_idx]
        
        # Urutkan berdasarkan skor tertinggi
        recommended_indices = product_scores.argsort()[::-1][1:6]  # Top 5 (exclude diri sendiri)
        recommended_products = df.iloc[recommended_indices]

        # Tampilkan tabel hasil rekomendasi
        st.write("Daftar Laptop yang Direkomendasikan:")
        st.dataframe(recommended_products[['name', 'price', 'rating']])  # Tampilkan kolom penting

        # Plot skor rekomendasi dalam bentuk bar chart
        st.subheader("Skor Rekomendasi")
        fig, ax = plt.subplots()
        ax.barh(recommended_products['name'], product_scores[recommended_indices], color='skyblue')
        ax.set_xlabel('Skor Kemiripan')
        ax.set_title('Top 5 Rekomendasi Laptop')
        st.pyplot(fig)
else:
    st.error("Data tidak memiliki kolom 'name'. Periksa file pickle.")
