import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_elements import elements, mui, html

# ========== CONFIG ==========
st.set_page_config(page_title="AR Dashboard", layout="wide")
st.title("📊 Customer AR Analytics Dashboard")
st.markdown("Visualize customer metrics with interactivity and style.")

# ========== LOAD CSV FROM GITHUB ==========
CSV_URL = "https://github.com/BhargavD22/Customer-Segmentation-with-Analytics-using-AR-/blob/b8891679954c60b12600fe072c164b515107e447/synthetic_ar_dataset_noisy.csv"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

try:
    df = load_data()
except Exception as e:
    st.error(f"Failed to load data. Please check the GitHub CSV URL.\n\n{e}")
    st.stop()

# ========== CUSTOMER SELECTION ==========
customer_list = df['CustomerID'].unique()
selected_customer = st.selectbox("Select a Customer", customer_list)

cust_data = df[df['CustomerID'] == selected_customer].iloc[0]

# ========== KPI ANIMATED TILES ==========
st.markdown("## 📌 Key Metrics")

with elements("metrics"):
    col1, col2, col3 = st.columns(3)
    with col1:
        with mui.Card(sx={"p": 3, "m": 1, "boxShadow": 3}):
            mui.Typography("Total Invoice Amount", variant="h6")
            mui.Typography(f"₹{cust_data['TotalInvoiceAmount']:,.2f}", variant="h4", color="primary")

    with col2:
        with mui.Card(sx={"p": 3, "m": 1, "boxShadow": 3}):
            mui.Typography("Total Outstanding", variant="h6")
            mui.Typography(f"₹{cust_data['TotalOutstanding']:,.2f}", variant="h4", color="error")

    with col3:
        with mui.Card(sx={"p": 3, "m": 1, "boxShadow": 3}):
            mui.Typography("Avg Payment Delay", variant="h6")
            mui.Typography(f"{cust_data['AvgPaymentDelay']:.2f} days", variant="h4", color="secondary")

st.markdown("")

with elements("more_metrics"):
    col4, col5 = st.columns(2)
    with col4:
        with mui.Card(sx={"p": 3, "m": 1, "boxShadow": 3}):
            mui.Typography("Payment Consistency", variant="h6")
            mui.Typography(f"{cust_data['PaymentConsistency']:.2f}", variant="h4", color="success")

    with col5:
        with mui.Card(sx={"p": 3, "m": 1, "boxShadow": 3}):
            mui.Typography("Response to Reminders", variant="h6")
            mui.Typography(f"{cust_data['ResponseToReminders']:.2f}", variant="h4", color="info")

# ========== COMPARISON CHARTS ==========
st.markdown("## 📉 Customer Comparison")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(df, x='CustomerID', y='AvgPaymentDelay',
                  color='CustomerID', title="Average Payment Delay by Customer")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df, x='CustomerID', y='TotalOutstanding',
                  color='CustomerID', title="Total Outstanding by Customer")
    st.plotly_chart(fig2, use_container_width=True)

st.caption("📎 Use this dashboard to identify payment behaviors and act early.")
