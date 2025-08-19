# ar_customer_dashboard.py

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
st.set_page_config(page_title="AR Customer Insights Dashboard", layout="wide")
st.title("👤 AR Customer-Centric Dashboard")

# ------------------
# LOAD DATA
# ------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload your AR dataset (.csv)", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head())

    # ------------------
    # FEATURE ENGINEERING
    # ------------------
    df['Avg_Payment_Delay'] = df.groupby('Customer_ID')['Payment_Delay_Days'].transform('mean')
    df['Total_Invoice_Amount'] = df.groupby('Customer_ID')['Invoice_Amount'].transform('sum')
    df['Total_Amount_Paid'] = df.groupby('Customer_ID')['Amount_Paid'].transform('sum')
    df['Total_Outstanding'] = df.groupby('Customer_ID')['Outstanding_Amount'].transform('sum')
    df['Payment_Consistency_Index'] = df.groupby('Customer_ID')['Payment_Consistency_Index'].transform('mean')
    df['Credit_Utilization_Velocity'] = df.groupby('Customer_ID')['Credit_Utilization_Velocity'].transform('mean')
    df['Response_to_Reminder_Ratio'] = df.groupby('Customer_ID')['Response_to_Reminder_Ratio'].transform('mean')

    # One row per customer
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

    # ------------------
    # CLUSTERING FOR CONTEXT
    # ------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df.drop(columns=['Customer_ID']))
    kmeans = KMeans(n_clusters=4, random_state=42)
    features_df['Segment'] = kmeans.fit_predict(X_scaled)

    # ------------------
    # CUSTOMER CENTRIC VIEW
    # ------------------
    st.markdown("---")
    st.header("🔎 Explore Individual Customers")

    customer_list = features_df['Customer_ID'].sort_values().unique()
    selected_customer = st.selectbox("Select a customer to explore:", customer_list)

    customer_info = features_df[features_df['Customer_ID'] == selected_customer]
    full_customer_txns = df[df['Customer_ID'] == selected_customer].sort_values(by='Invoice_Date')

    st.subheader(f"📌 Key Metrics for Customer: {selected_customer}")
    st.metric("Total Invoice Amount", f"₹{float(customer_info['Total_Invoice_Amount']):,.2f}")
    st.metric("Total Outstanding", f"₹{float(customer_info['Total_Outstanding']):,.2f}")
    st.metric("Avg Payment Delay", f"{float(customer_info['Avg_Payment_Delay']):.2f} days")
    st.metric("Payment Consistency", f"{float(customer_info['Payment_Consistency_Index']):.2f}")
    st.metric("Response to Reminders", f"{float(customer_info['Response_to_Reminder_Ratio']):.2f}")

    # ------------------
    # VISUALS
    # ------------------
    st.subheader("📈 Customer Transaction History")
    fig = px.line(full_customer_txns, x='Invoice_Date', y='Outstanding_Amount', title="Outstanding Over Time")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(full_customer_txns, x='Invoice_Date', y='Invoice_Amount', title="Invoices Raised")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🧾 All Transactions for Selected Customer")
    st.dataframe(full_customer_txns.reset_index(drop=True))

    st.markdown("---")
    st.success("This dashboard provides customer-level AR insights including payment behavior, risk profiling, and engagement.")

else:
    st.warning("📂 Please upload a valid CSV file to proceed.")
    st.markdown("You can use the synthetic dataset generated previously.")
