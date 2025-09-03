# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from google.cloud import bigquery
from google.oauth2 import service_account
from fpdf import FPDF
import io
import os

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Customer Segmentation with AR Analytics", layout="wide")
st.title("📊 Customer Segmentation with AR Analytics")

# ==============================
# DATA LOADING
# ==============================
def load_data_from_bigquery():
    credentials = service_account.Credentials.from_service_account_info(
        dict(st.secrets["bigquery"])   # ✅ use your current [bigquery] section
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    query = """
        SELECT *
        FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
    """
    return client.query(query).to_dataframe()

@st.cache_data
def get_customer_summary(df):
    summary = df.groupby("Customer_ID").agg({
        "Invoice_Amount": "sum",
        "Outstanding_Amount": "sum",
        "Payment_Delay_Days": "mean",
        "Payment_Consistency_Index": "mean",
        "Response_to_Reminder_Ratio": "mean",
        "High_Risk_Flag": "max"
    }).reset_index()

    # Rule-based risk
    summary["Rule_Based_Risk"] = summary["High_Risk_Flag"].apply(lambda x: "🔴 High" if x == 1 else "🟢 Low")

    # ML clustering
    features = summary[["Invoice_Amount", "Outstanding_Amount", "Payment_Delay_Days",
                        "Payment_Consistency_Index", "Response_to_Reminder_Ratio"]].fillna(0)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    summary["Cluster"] = kmeans.fit_predict(scaled)

    # Map clusters to risk labels
    risk_map = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
    summary["ML_Risk"] = summary["Cluster"].map(risk_map)

    return summary

# ==============================
# PDF GENERATION
# ==============================
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()

    font_path = os.path.join("fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 14)
    pdf.cell(200, 10, txt="📊 Customer AR Summary Report", ln=True, align="C")

    pdf.set_font("DejaVu", "", 12)
    for _, row in data.iterrows():
        pdf.ln(5)
        pdf.cell(200, 10,
                 txt=f"🧑 Customer: {row['Customer_ID']} | Rule Risk: {row['Rule_Based_Risk']} | ML Risk: {row['ML_Risk']}",
                 ln=True)
        pdf.cell(200, 10,
                 txt=f"💰 Total Invoice: ₹{row['Invoice_Amount']:,.0f}, 📌 Outstanding: ₹{row['Outstanding_Amount']:,.0f}",
                 ln=True)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

# ==============================
# MAIN APP
# ==============================
with st.spinner("Fetching data from BigQuery..."):
    df = load_data_from_bigquery()

st.success("✅ Data successfully loaded from BigQuery")

# Summary metrics
st.subheader("📌 Key Business KPIs")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("💰 Total Invoice", f"₹ {df['Invoice_Amount'].sum():,.0f}")
col2.metric("📉 Total Outstanding", f"₹ {df['Outstanding_Amount'].sum():,.0f}")
col3.metric("⏱️ Avg Payment Delay", f"{df['Payment_Delay_Days'].mean():.2f} days")
col4.metric("📈 Avg Consistency Index", f"{df['Payment_Consistency_Index'].mean():.2f}")
col5.metric("📬 Avg Response Ratio", f"{df['Response_to_Reminder_Ratio'].mean():.2f}")

st.markdown("---")

# Customer Summary
st.subheader("📊 Customer Segmentation & Risk Overview")
customer_summary = get_customer_summary(df)
st.dataframe(customer_summary)

# Scatter Plot
st.subheader("👥 Customer Comparison")
fig = px.scatter(customer_summary,
                 x="Payment_Delay_Days", y="Outstanding_Amount",
                 color="ML_Risk", size="Invoice_Amount",
                 hover_name="Customer_ID",
                 title="Outstanding vs Delay with ML Risk Clusters")
st.plotly_chart(fig, use_container_width=True)

# Individual Customer View
st.subheader("🔍 Individual Customer View")
selected_customer = st.selectbox("Select Customer ID", customer_summary["Customer_ID"].unique())

if selected_customer:
    cust_data = df[df["Customer_ID"] == selected_customer]
    cust_summary = customer_summary[customer_summary["Customer_ID"] == selected_customer]

    st.write("### Customer Metrics")
    st.metric("Total Invoice Amount", f"₹ {cust_summary['Invoice_Amount'].values[0]:,.0f}")
    st.metric("Total Outstanding", f"₹ {cust_summary['Outstanding_Amount'].values[0]:,.0f}")
    st.metric("Avg Payment Delay", f"{cust_summary['Payment_Delay_Days'].values[0]:.2f} days")
    st.metric("Consistency Index", f"{cust_summary['Payment_Consistency_Index'].values[0]:.2f}")
    st.metric("Reminder Response Ratio", f"{cust_summary['Response_to_Reminder_Ratio'].values[0]:.2f}")
    st.metric("Rule-Based Risk", cust_summary["Rule_Based_Risk"].values[0])
    st.metric("ML Cluster Risk", cust_summary["ML_Risk"].values[0])

    st.write("### Invoices")
    st.dataframe(cust_data)

# Export PDF
st.subheader("📤 Export Reports")
pdf_file = generate_pdf(customer_summary)
st.download_button("📄 Download PDF Report", data=pdf_file, file_name="customer_summary.pdf")
