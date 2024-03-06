import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data
@st.cache
def load_data():
    data = pd.read_csv("data/day.csv")
    return data

data = load_data()

# Title
st.title('Dashboard Analisis Data: Bike Sharing Dataset')

# Sidebar
st.sidebar.header('Pertanyaan Bisnis')
question = st.sidebar.selectbox('Pilih Pertanyaan Bisnis:', ('Pertanyaan 1', 'Pertanyaan 2'))

# Data Wrangling
## Cleaning Data
Q1 = (data['hum']).quantile(0.25)
Q3 = (data['hum']).quantile(0.75)
IQR = Q3 - Q1
 
maximum = Q3 + (1.5*IQR)
minimum = Q1 - (1.5*IQR)
 
kondisi_lower_than = data['hum'] < minimum
kondisi_more_than = data['hum'] > maximum
 
data.loc[data['hum'] > maximum, 'hum'] = maximum
data.loc[data['hum'] < minimum, 'hum'] = minimum

# Exploratory Data Analysis
if question == 'Pertanyaan 1':
    st.subheader('Pertanyaan 1: Bagaimana penggunaan sepeda berbanding dengan suhu dan kelembaban?')
    
    # Korelasi antara suhu dan jumlah pengguna sepeda
    correlation_temp_cnt = data[['temp', 'cnt']].corr().loc['cnt', 'temp']
    st.write("Korelasi antara suhu dan jumlah pengguna sepeda:", correlation_temp_cnt)
    
    # Korelasi antara kelembaban dan jumlah pengguna sepeda
    correlation_hum_cnt = data[['hum', 'cnt']].corr().loc['cnt', 'hum']
    st.write("Korelasi antara kelembaban dan jumlah pengguna sepeda:", correlation_hum_cnt)
    
    # Scatter plot untuk hubungan suhu dan jumlah pengguna sepeda
    st.write("Scatter Plot: Hubungan Suhu dan Jumlah Pengguna Sepeda")
    scatter = plt.scatter(data['temp'], data['cnt'], c=data['hum'], cmap='viridis', alpha=0.5)
    plt.xlabel('Suhu (temp)')
    plt.ylabel('Jumlah Pengguna Sepeda (cnt)')
    plt.title('Scatter Plot: Hubungan Suhu dan Jumlah Pengguna Sepeda')
    plt.colorbar(scatter, label='Humidity')
    st.pyplot()

elif question == 'Pertanyaan 2':
    st.subheader('Pertanyaan 2: Bagaimana cuaca memengaruhi penggunaan sepeda?')
    
    # Korelasi antara cuaca dan jumlah pengguna sepeda
    correlation_weathersit_cnt = data['weathersit'].corr(data['cnt'])
    st.write("Korelasi cuaca dan jumlah pengguna sepeda:", correlation_weathersit_cnt)
    
    # Scatter plot untuk hubungan cuaca dan jumlah pengguna sepeda
    st.write("Visualisasi scatter plot untuk cuaca dan jumlah pengguna sepeda")
    scatter = plt.scatter(data['weathersit'], data['cnt'], alpha=0.5)
    plt.xlabel('cuaca')
    plt.ylabel('Jumlah pengguna sepeda')
    plt.title('Visualisasi scatter plot untuk cuaca dan jumlah pengguna sepeda')
    st.pyplot(scatter)

    # K-means Clustering
    st.subheader('Analisis Clustering Menggunakan K-means')
    
    # Pilih fitur yang akan digunakan untuk clustering
    X = data[['temp', 'hum', 'weathersit', 'cnt']]

    # Tentukan jumlah cluster yang diinginkan
    n_clusters = 3

    # Terapkan algoritma K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)

    # Dapatkan label cluster untuk setiap sampel data
    labels = kmeans.labels_

    # Evaluasi hasil clustering, misalnya dengan menggunakan silhouette score
    silhouette_avg = silhouette_score(X, labels)
    st.write("Silhouette Score:", silhouette_avg)

    # Visualisasi hasil clustering
    st.write("Visualisasi hasil clustering")
    scatter = plt.scatter(X['temp'], X['cnt'], c=labels, cmap='viridis', alpha=0.5)
    plt.xlabel('Suhu (temp)')
    plt.ylabel('Jumlah Pengguna Sepeda (cnt)')
    plt.colorbar(scatter, label='Cluster')
    plt.title('K-means Clustering: Hubungan Suhu dan Jumlah Pengguna Sepeda')
    st.pyplot()
