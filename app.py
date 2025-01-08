pip install seaborn matplotlib pandas numpy scikit-learn streamlit
import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan data
with open('laptop_recommender.pkl', 'rb') as f:
    recommender = pickle.load(f)

# Judul aplikasi
st.title("💻 Sistem Rekomendasi Laptop E-commerce")
st.markdown("""
Aplikasi ini menggunakan metode **Collaborative Filtering dengan Cosine Similarity** 
untuk merekomendasikan laptop berdasarkan produk yang dipilih. 
""")

# Dropdown untuk memilih produk
product_name = st.selectbox('Pilih Laptop', recommender.df['name'], index=0)

# Menampilkan rekomendasi produk
if product_name:
    st.subheader(f"Rekomendasi untuk '{product_name}'")
    recommended_products = recommender.recommend(product_name, topk=5)

    # Tampilkan tabel hasil rekomendasi
    st.write("Daftar Laptop yang Direkomendasikan:")
    st.dataframe(recommended_products)

    # Plot skor rekomendasi dalam bentuk bar chart
    st.subheader("Skor Rekomendasi")
    fig, ax = plt.subplots()
    ax.barh(recommended_products['name'], np.arange(5, 0, -1), color='skyblue')
    ax.set_xlabel('Skor')
    ax.set_ylabel('Laptop')
    ax.set_title('Top 5 Rekomendasi Laptop')
    st.pyplot(fig)

    # Visualisasi matriks kemiripan (Heatmap)
    st.subheader("Matriks Kemiripan Produk")
    similarity_matrix = cosine_similarity(recommender.bank)
    similarity_df = pd.DataFrame(similarity_matrix, 
                                 index=recommender.df['name'], 
                                 columns=recommender.df['name'])
    plt.figure(figsize=(12, 8))
    sns.heatmap(similarity_df, cmap='coolwarm', xticklabels=False, yticklabels=False)
    plt.title('Heatmap Matriks Kemiripan')
    st.pyplot(plt)
