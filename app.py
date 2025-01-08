import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan data
@st.cache_resource
def load_recommender():
    with open('laptop_recommender.pkl', 'rb') as file:
        recommender = pickle.load(file)
    return recommender

# Load data
recommender = load_recommender()
df = recommender['data']  # Asumsi file pickle memiliki data dalam key 'data'
similarity_matrix = recommender['similarity_matrix']  # Matriks kemiripan

# Judul aplikasi
st.title("💻 Sistem Rekomendasi Laptop E-commerce")
st.markdown("""
Aplikasi ini menggunakan metode **Collaborative Filtering dengan Cosine Similarity** 
untuk merekomendasikan laptop berdasarkan produk yang dipilih. 
""")

# Dropdown untuk memilih produk
product_name = st.selectbox('Pilih Laptop', df['name'])

# Menampilkan rekomendasi produk
if product_name:
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

    # Visualisasi matriks kemiripan (Heatmap)
    st.subheader("Matriks Kemiripan Produk")
    plt.figure(figsize=(10, 6))
    sns.heatmap(similarity_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title('Heatmap Matriks Kemiripan Produk')
    st.pyplot(plt)
