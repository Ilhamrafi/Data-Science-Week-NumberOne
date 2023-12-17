import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import seaborn as sns
from train import *


selected = option_menu(None, ["Dashboard", "Pattern Analysis"], 
    icons=['archive', 'cloud-upload'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected == 'Dashboard':
    st.title("Dataset :")
    st.markdown("Kumpulan data berisi data informasi tentang pelanggan dari sebuah perusahaan telekomunikasi. Data ini mencakup berbagai atribut, termasuk informasi tentang pelanggan, penggunaan layanan, lokasi pelanggan, dan metode pembayaran. :")
    
    # Load your DataFrame (df_train) here, replace with the actual data source
    df_train = pd.read_excel("Telco_customer_churn_adapted_v2.xlsx")
    st.write(df_train)
    
    st.title('EDA (Exploratory Data Analysis)')

    # Visualisasi 1: Jumlah Pelanggan Berdasarkan Lokasi
    st.markdown('Jumlah Pelanggan Berdasarkan Lokasi :')
    customer_count_by_location = df_train['Location'].value_counts()
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = plt.barh(customer_count_by_location.index, customer_count_by_location.values, color=['steelblue', 'indianred', 'sandybrown'], edgecolor='black')
    plt.title('Jumlah Pelanggan Berlangganan Berdasarkan Lokasi')
    plt.xlabel('Jumlah Pelanggan')
    plt.ylabel('Lokasi')

    for bar in bars:
        yval = bar.get_width()
        plt.text(yval, bar.get_y() + bar.get_height()/2, round(yval), va='center', ha='left', color='black', fontweight='bold')

    # Button expander untuk visualisasi 1
    with st.expander("Tampilkan Visualisasi Lokasi"):
        st.pyplot(fig)

    # Button expander untuk visualisasi histogram Tenure Months Visualisasi 2
    st.markdown('Distribusi Tenure Months :')
    with st.expander("Tampilkan Distribusi Tenure Months"):
        tenure_stats = df_train['Tenure Months'].describe()

        plt.figure(figsize=(10, 6))
        sns.histplot(df_train['Tenure Months'], bins=30, kde=True, color='crimson')
        plt.title('Distribusi Tenure Months', fontsize=16, fontweight='bold')
        plt.xlabel('Tenure Months', fontsize=12)
        plt.ylabel('Jumlah Pelanggan', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(plt.gcf())  # Menampilkan plot histogram di Streamlit
    
    # Data Visualisasi 3
    rentang_tenure = ['0-12 bulan', '13-24 bulan', '25-36 bulan', '37-48 bulan', '49-60 bulan', '61-72 bulan']
    jumlah_pelanggan = [2186, 1024, 832, 762, 832, 1407]
    colors = ['grey', 'indianred', 'sandybrown', 'darkcyan', 'indigo', 'lightgreen']

    st.markdown('Visualisasi Jumlah Pelanggan dalam Rentang Tenure :')
    # Button expander untuk visualisasi rentang Tenure
    with st.expander("Tampilkan Jumlah Pelanggan dalam Rentang Tenure"):
        plt.figure(figsize=(10, 6))
        bars = plt.barh(rentang_tenure, jumlah_pelanggan, color=colors, edgecolor='black')

        for bar in bars:
            xval = bar.get_width()
            plt.text(xval + 10, bar.get_y() + bar.get_height()/2, round(xval, 2), va='center', color='black', fontweight='bold')

        plt.title('Jumlah Pelanggan dalam Rentang Tenure')
        plt.ylabel('Rentang Tenure')
        plt.xlabel('Jumlah Pelanggan')

        st.pyplot(plt.gcf())  # Menampilkan plot histogram di Streamlit

    # Visualisasi 4: Jumlah Pelanggan Berdasarkan Device Class
    st.markdown('Jumlah Pelanggan Berdasarkan Device Class :')
    customer_count_by_device_class = df_train['Device Class'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bars2 = plt.barh(customer_count_by_device_class.index, customer_count_by_device_class.values, color=['steelblue', 'indianred', 'sandybrown'], edgecolor='black')

    for bar in bars2:
        xval = bar.get_width()
        plt.text(xval + 10, bar.get_y() + bar.get_height()/2, round(xval, 2), va='center', color='black', fontweight='bold')

    plt.title('Jumlah Pelanggan Berdasarkan Device Class')
    plt.xlabel('Jumlah Pelanggan')
    plt.ylabel('Device Class')

    with st.expander("Tampilkan Visualisasi Device Class"):
        st.pyplot(fig2)

    # Data Visualisasi 5 (real)
    products = ['Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp']
    no_counts = [2257, 1753, 1739, 2230, 1346, 1345]
    yes_counts = [839, 1343, 1357, 866, 1750, 1751]

    st.markdown('Visualisasi Penggunaan Produk dan Layanan :')
    # Button expander untuk visualisasi produk dan layanan
    with st.expander("Tampilkan Penggunaan Produk dan Layanan"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(products, no_counts, width=0.4, align='center', label='No', color='steelblue', edgecolor='black')
        ax.bar(products, yes_counts, width=0.4, align='edge', label='Yes', color='indianred', edgecolor='black')

        plt.plot(products, no_counts, marker='o', color='blue', label='No Line')
        plt.plot(products, yes_counts, marker='o', color='red', label='Yes Line')

        ax.legend()
        ax.set_xlabel('Produk / Layanan')
        ax.set_ylabel('Jumlah Penggunaan')
        ax.set_title('Frekuensi Penggunaan Produk dan Layanan yang Berbeda Untuk Pelanggan High End')
        plt.xticks(rotation=45)

        st.pyplot(plt.gcf())  # Menampilkan plot di Streamlit

    # Data Visualisasi 6 (real)
    products = ['Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp']
    no_counts = [1241, 1335, 1356, 1243, 1464, 1440]
    yes_counts = [1180, 1086, 1065, 1178, 957, 981]

    st.markdown('Visualisasi Frekuensi Penggunaan Produk dan Layanan untuk Pelanggan Mid End :')
    # Button expander untuk visualisasi produk dan layanan pelanggan Mid End
    with st.expander("Tampilkan Frekuensi Penggunaan Produk dan Layanan Pelanggan Mid End"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(products, no_counts, width=0.4, align='center', label='No', color='steelblue', edgecolor='black')
        ax.bar(products, yes_counts, width=0.4, align='edge', label='Yes', color='indianred', edgecolor='black')

        plt.plot(products, no_counts, marker='o', color='blue', label='No Line')
        plt.plot(products, yes_counts, marker='o', color='red', label='Yes Line')

        ax.legend()
        ax.set_xlabel('Produk / Layanan')
        ax.set_ylabel('Jumlah Penggunaan')
        ax.set_title('Frekuensi Penggunaan Produk dan Layanan yang Berbeda Untuk Pelanggan Mid End')
        plt.xticks(rotation=45)

        st.pyplot(plt.gcf())  # Menampilkan plot di Streamlit
    
    # Data Visualisasi 7 (Real)
    products = ['Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp']
    usage_counts = [1526, 1526, 1526, 1526, 1526, 1526]

    st.markdown('Visualisasi Frekuensi Penggunaan Produk dan Layanan untuk Pelanggan Low End :')
    # Button expander untuk visualisasi produk dan layanan pelanggan Low End
    with st.expander("Tampilkan Frekuensi Penggunaan Produk dan Layanan Pelanggan Low End"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(products, usage_counts, color='steelblue', label='Usage Count', edgecolor='black')
        ax.plot(products, usage_counts, marker='o', color='red', label='Usage Count (line)')

        ax.legend()
        ax.set_xlabel('Produk / Layanan')
        ax.set_ylabel('Jumlah Penggunaan')
        ax.set_title('Frekuensi Penggunaan Produk dan Layanan yang Berbeda Untuk Pelanggan Low End')
        plt.xticks(rotation=45)

        st.pyplot(plt.gcf())  # Menampilkan plot di Streamlit

    # Data Visualisasi 8 (Real)
    device_classes = ['High End', 'Low End', 'Mid End']
    monthly_purchase = [118.950168, 27.402952, 75.532819]

    st.markdown('Visualisasi Rata-rata Pembelian Bulanan Berdasarkan Device Class :')
    # Button expander untuk visualisasi rata-rata pembelian bulanan berdasarkan Device Class
    with st.expander("Tampilkan Rata-rata Pembelian Bulanan Berdasarkan Device Class"):
        # Membuat DataFrame
        df = pd.DataFrame({'Device Class': device_classes, 'Monthly Purchase (Thou. IDR)': monthly_purchase})
        # Mengurutkan DataFrame berdasarkan Monthly Purchase
        df = df.sort_values(by='Monthly Purchase (Thou. IDR)', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df['Device Class'], df['Monthly Purchase (Thou. IDR)'], color=['steelblue', 'indianred', 'sandybrown'], edgecolor='black')

        # Menambahkan label di sekitar batang
        for bar in bars:
            xval = bar.get_width()
            ax.text(xval + 0.1, bar.get_y() + bar.get_height() / 2, round(xval, 2), va='center', color='black', fontweight='bold')

        ax.set_title('Rata-rata Pembelian Bulanan Berdasarkan Device Class')
        ax.set_xlabel('Rata-rata Pembelian Bulanan (Thou. IDR)')
        ax.set_ylabel('Device Class')

        st.pyplot(fig)  # Menampilkan plot di Streamlit

    # Menghitung jumlah penggunaan setiap metode pembayaran Visualisasi 9 (Real)
    payment_method_counts = df_train['Payment Method'].value_counts()

    st.markdown('Visualisasi Jumlah Penggunaan Metode Pembayaran :')

    # Button expander untuk visualisasi jumlah penggunaan metode pembayaran
    with st.expander("Tampilkan Jumlah Penggunaan Metode Pembayaran"):
        # Membuat bar plot menyamping
        plt.figure(figsize=(10, 6))
        bars = plt.barh(payment_method_counts.index, payment_method_counts.values, color=['steelblue', 'sandybrown', 'lightgreen', 'indianred'], edgecolor='black')

        # Menambahkan label di sebelah kanan batang
        for bar in bars:
            xval = bar.get_width()
            plt.text(xval + 10, bar.get_y() + bar.get_height()/2, round(xval, 2), va='center', ha='left', color='black', fontweight='bold')

        plt.title('Jumlah Penggunaan Metode Pembayaran')
        plt.xlabel('Jumlah Penggunaan')
        plt.ylabel('Metode Pembayaran')

        st.pyplot(plt)  # Menampilkan plot di Streamlit

    # Visualisasi 10 (Real)
    st.markdown('Visualisasi Distribusi CLTV :')
    # Button expander untuk visualisasi distribusi CLTV
    with st.expander("Tampilkan Distribusi CLTV"):
        plt.figure(figsize=(10, 6))
        sns.histplot(df_train['CLTV (Predicted Thou. IDR)'], bins=30, color='crimson', kde=True)
        plt.title('Distribusi CLTV')
        plt.xlabel('CLTV (Predicted Thou. IDR)')
        plt.ylabel('Jumlah Pelanggan')

        st.pyplot(plt)  # Menampilkan plot di Streamlit

    # Visualisasi 11 (Real)
    st.markdown('Persentase Churn :')
    labels = ['No', 'Yes']
    persentase = [73.463013, 26.536987]
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = plt.pie(persentase, labels=labels, autopct='%1.1f%%', colors=['steelblue', 'indianred'], startangle=140, wedgeprops={'edgecolor': 'black'})
    plt.title('Persentase Churn')

    # Button expander untuk visualisasi 3
    with st.expander("Tampilkan Persentase Churn (Pie Chart)"):
        st.pyplot(fig3)

    # Visualisasi 12 (Real)
    st.markdown('Jumlah Churn Berdasarkan Device Class :')
    churn_data = {
        'Churn Label': ['No', 'Yes'],
        'High End': [1799, 1297],
        'Mid End': [1962, 459],
        'Low End': [1413, 113]
    }
    df = pd.DataFrame(churn_data)
    df.set_index('Churn Label', inplace=True)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    colors = ['indianred', 'steelblue', 'sandybrown']
    df.plot(kind='barh', ax=ax4, color=colors, edgecolor='black')

    plt.title('Jumlah Churn Berdasarkan Device Class')
    plt.xlabel('Jumlah Churn')
    plt.ylabel('Device Class')
    plt.xticks(rotation=0)

    for i, value in enumerate(df.columns):
        for index, val in enumerate(df[value]):
            ax4.text(val, index, str(val), ha='left', va='center', color='black', fontweight='bold')

    # Button expander untuk visualisasi 
    with st.expander("Tampilkan Jumlah Churn Berdasarkan Device Class (Bar Plot)"):
        st.pyplot(fig4)
    

    # Visualisasi 13
    churn_data = {
        'Churn Label': ['No', 'Yes'],
        'High End': [1799, 1297],
        'Mid End': [1962, 459],
        'Low End': [1413, 113]
    }

    # Membuat DataFrame dari data
    df = pd.DataFrame(churn_data)
    df.set_index('Churn Label', inplace=True)

if selected == 'Pattern Analysis':
    st.title("Pattern Analysis")
    eval, y_train, feature_importance= train()
    eval_smote, y_train_resampled, feature_importance_smote = train_smote()
    
    st.write("Informasi Data sebelum dan sesudah SMOTE:")
    visualisasi_label(y_train, y_train_resampled)
    st.write("Perbandingan Model Machine Learning Sebelum SMOTE")
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            st.write(eval)
        with col_2:
            visualisasi_perbadingan(eval)
    plot_feature_importance(feature_importance)
    st.write("Perbandingan Model Machine Learning Sesudah SMOTE")
    with st.container():
        col_1, col_2 = st.columns(2)
        with col_1:
            st.write(eval_smote)
        with col_2:
            visualisasi_perbadingan(eval_smote)
    plot_feature_importance(feature_importance_smote)
