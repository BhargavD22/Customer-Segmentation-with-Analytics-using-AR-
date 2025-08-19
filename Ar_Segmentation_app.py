# ar_segmentation_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# ------------------
# PAGE SETUP
# ------------------
st.set_page_config(page_title="AR Customer Segmentation", layout="wide")
st.title("📊 AR Customer Segmentation Dashboard")

# ------------------
# LOAD DATA
# ------------------
st.sidebar.header("Step 1: Upload Your Dataset")
file = st.sidebar.file_uploader("Upload AR Dataset (.csv)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head())

    # ------------------
    # FEATURE ENGINEERING
    # ------------------
    st.sidebar.header("Step 2: Feature Selection")
    st.markdown("---")
    
    # Derive or clean needed columns
    df['Avg_Payment_Delay'] = df.groupby('Customer_ID')['Payment_Delay_Days'].transform('mean')
    df['Total_Invoice_Amount'] = df.groupby('Customer_ID')['Invoice_Amount'].transform('sum')
    df['Total_Amount_Paid'] = df.groupby('Customer_ID')['Amount_Paid'].transform('sum')
    df['Total_Outstanding'] = df.groupby('Customer_ID')['Outstanding_Amount'].transform('sum')
    df['Payment_Consistency_Index'] = df.groupby('Customer_ID')['Payment_Consistency_Index'].transform('mean')
    df['Credit_Utilization_Velocity'] = df.groupby('Customer_ID')['Credit_Utilization_Velocity'].transform('mean')
    df['Response_to_Reminder_Ratio'] = df.groupby('Customer_ID')['Response_to_Reminder_Ratio'].transform('mean')

    # Drop duplicates to keep one row per customer
    features_df = df.drop_duplicates(subset='Customer_ID')[[
        'Customer_ID',
        'Avg_Payment_Delay',
        'Total_Invoice_Amount',
        'Total_Amount_Paid',
        'Total_Outstanding',
        'Payment_Consistency_Index',
        'Credit_Utilization_Velocity',
        'Response_to_Reminder_Ratio'
    ]].dropna()

    st.subheader("Selected Features per Customer")
    st.dataframe(features_df.head())

    # ------------------
    # CLUSTERING (K-MEANS)
    # ------------------
    st.sidebar.header("Step 3: Choose Clustering Settings")
    num_clusters = st.sidebar.slider("Select number of clusters:", 2, 10, 4)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.drop(columns=['Customer_ID']))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    features_df['Segment'] = kmeans.fit_predict(X_scaled)

    # ------------------
    # PCA FOR VISUALIZATION
    # ------------------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    features_df['PCA1'] = X_pca[:, 0]
    features_df['PCA2'] = X_pca[:, 1]

    # ------------------
    # SEGMENTATION PLOT
    # ------------------
    st.subheader("📌 Customer Segments (PCA View)")
    fig = px.scatter(features_df, x='PCA1', y='PCA2', color=features_df['Segment'].astype(str),
                     hover_data=['Customer_ID'], title="Customer Clusters (2D PCA)")
    st.plotly_chart(fig, use_container_width=True)

    # ------------------
    # KPI SECTION
    # ------------------
    st.markdown("---")
    st.subheader("📈 Business KPIs by Segment")

    kpi_df = features_df.groupby('Segment').agg({
        'Total_Outstanding': 'mean',
        'Avg_Payment_Delay': 'mean',
        'Payment_Consistency_Index': 'mean',
        'Response_to_Reminder_Ratio': 'mean'
    }).reset_index()

    st.dataframe(kpi_df.style.format("{:.2f}"))

    # ------------------
    # SEGMENT DETAILS
    # ------------------
    st.markdown("---")
    st.subheader("🔍 Segment Profiles")
    selected_segment = st.selectbox("Select a segment to view its customers", sorted(features_df['Segment'].unique()))
    st.dataframe(features_df[features_df['Segment'] == selected_segment])

else:
    st.warning("📂 Please upload a CSV file to proceed.")
    st.markdown("You can use the synthetic dataset we generated earlier for this AR segmentation use case.")
