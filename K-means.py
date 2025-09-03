# app.py

import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.express as px

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Customer AR Insights Dashboard", layout="wide")

# ==============================
# COMPANY LOGO + TITLE
# ==============================
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:15px; margin-bottom:20px;">
        <img src="miracle-logo-dark.png" alt="Miracle Logo" width="220">
        <h1 style="color:#333; margin:0;">Customer AR Insights Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Data is securely fetched from **BigQuery** or uploaded via **CSV**.")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data_from_bigquery():
    from google.cloud import bigquery
    from google.oauth2 import service_account

    # Authenticate
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)

    
    query = """
        SELECT *
        FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
    """
    df = client.query(query).to_dataframe()
    return df

# File uploader option
data_source = st.radio("Select Data Source:", ["BigQuery", "Upload CSV"], horizontal=True)

if data_source == "BigQuery":
    df = load_data_from_bigquery()
elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()
else:
    st.stop()

# ==============================
# KPI METRICS (Minimalist Style)
# ==============================
st.markdown("### 📌 Key Business KPIs")

def kpi_card(title, value, unit="", threshold=None, higher_is_bad=True):
    if threshold is not None:
        if higher_is_bad:
            color = "#FF4B4B" if value > threshold else "#2ECC71"  # red/green
        else:
            color = "#2ECC71" if value > threshold else "#FF4B4B"
    else:
        color = "#00AEEF"  # Miracle blue

    return f"""
        <div style="background-color:white; border:1px solid #e0e0e0;
                    border-radius:15px; padding:20px; margin:10px;
                    text-align:center; box-shadow:0px 4px 10px rgba(0,0,0,0.05);">
            <h4 style="margin:0; color:#333; font-weight:600;">{title}</h4>
            <p style="margin:5px 0; font-size:22px; font-weight:bold; color:{color};">
                {value:,.2f}{unit}
            </p>
        </div>
    """

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(kpi_card("👥 Total Customers", df["Customer_ID"].nunique(), unit="", threshold=1000, higher_is_bad=False), unsafe_allow_html=True)

with col2:
    st.markdown(kpi_card("💰 Total Outstanding", df["Outstanding_Amount"].sum(), unit=" ₹", threshold=500000), unsafe_allow_html=True)

with col3:
    st.markdown(kpi_card("⏱ Avg Payment Delay", df["Payment_Delay_Days"].mean(), unit=" days", threshold=30), unsafe_allow_html=True)

with col4:
    st.markdown(kpi_card("⚠️ High Risk Customers", df[df["High_Risk_Flag"]==1]["Customer_ID"].nunique(), unit="", threshold=50), unsafe_allow_html=True)

# ==============================
# CUSTOMER SELECTION + GRAPH
# ==============================
st.markdown("### 📊 Customer Insights")

selected_customer = st.selectbox("Select Customer", df["Customer_ID"].unique())

customer_data = df[df["Customer_ID"] == selected_customer]

# Revenue Trend Graph
fig = px.line(
    customer_data,
    x="Invoice_Date",
    y="Invoice_Amount",
    title=f"Revenue Trend for Customer {selected_customer}",
    markers=True,
    line_shape="spline",
)
fig.update_traces(line=dict(color="#00AEEF"))  # Miracle blue
st.plotly_chart(fig, use_container_width=True)

# ==============================
# CUSTOMER DETAILS TABLE
# ==============================
st.markdown("### 📑 Customer Details")
st.dataframe(customer_data, use_container_width=True)

# ==============================
# FOOTER WATERMARK
# ==============================
st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align:center; opacity:0.7;">
        <img src="miracle-logo-dark.png" alt="Miracle Logo" width="140">
        <p style="margin-top:5px; font-size:13px; color:#555;">
            © 2025 Miracle Software Systems - All Rights Reserved
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
