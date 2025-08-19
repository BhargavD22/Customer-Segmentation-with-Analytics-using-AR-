# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

# ========== EMBEDDED DATA ==========
@st.cache_data
def load_data():
    # Normally loaded from CSV, here we embed directly
    data = {
        'CustomerID': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
        'TotalInvoiceAmount': [1023890.75, 923450.60, 755430.40, 880125.30, 812989.51],
        'TotalOutstanding': [523890.35, 430120.10, 240360.25, 410555.75, 380353.21],
        'AvgPaymentDelay': [32.5, 27.8, 25.3, 28.4, 30.61],
        'PaymentConsistency': [25.0, 29.1, 32.7, 28.2, 27.57],
        'ResponseToReminders': [0.45, 0.65, 0.72, 0.48, 0.55]
    }
    df = pd.DataFrame(data)
    return df

df = load_data()

# ========== STREAMLIT PAGE CONFIG ==========
st.set_page_config(page_title="AR Customer Insights", layout="centered")

# ========== HEADER ==========
st.markdown("## 📊 Customer AR Analytics Dashboard")
st.markdown("Visualize and understand customer behavior based on AR data.")

# ========== CUSTOMER SELECTION ==========
customer_list = df['CustomerID'].unique()
selected_customer = st.selectbox("Select a Customer", customer_list)

cust_data = df[df['CustomerID'] == selected_customer].iloc[0]

# ========== CUSTOMER KPI METRICS ==========
st.markdown(f"### 📌 Key Metrics for Customer: `{selected_customer}`")

st.metric("Total Invoice Amount", f"₹{cust_data['TotalInvoiceAmount']:,.2f}")
st.metric("Total Outstanding", f"₹{cust_data['TotalOutstanding']:,.2f}")
st.metric("Avg Payment Delay", f"{cust_data['AvgPaymentDelay']:.2f} days")
st.metric("Payment Consistency", f"{cust_data['PaymentConsistency']:.2f}")
st.metric("Response to Reminders", f"{cust_data['ResponseToReminders']:.2f}")

st.divider()

# ========== CUSTOMER VS OTHERS ==========
st.subheader("📉 Comparison with Other Customers")

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(df, x='CustomerID', y='AvgPaymentDelay',
                  color='CustomerID', title="Average Payment Delay")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.bar(df, x='CustomerID', y='TotalOutstanding',
                  color='CustomerID', title="Total Outstanding")
    st.plotly_chart(fig2, use_container_width=True)

st.caption("📎 This tool can help stakeholders segment customers and act early on payment delays.")
