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
        dict(st.secrets["bigquery"])
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
# HELPER: CARD STYLING
# ==============================
def style_card(value, thresholds, colors):
    if value >= thresholds[0]:
        color = colors[0]   # red
    elif value >= thresholds[1]:
        color = colors[1]   # yellow
    else:
        color = colors[2]   # green
    return f"background-color:{color}; padding:20px; border-radius:12px; text-align:center; font-size:18px;"

# ==============================
# MAIN APP
# ==============================
with st.spinner("Fetching data from BigQuery..."):
    df = load_data_from_bigquery()

st.success("✅ Data successfully loaded from BigQuery")

# ==============================
# BUSINESS KPIs (Stylish)
# ==============================
st.subheader("📌 Key Business KPIs")

total_invoice = df['Invoice_Amount'].sum()
total_outstanding = df['Outstanding_Amount'].sum()
avg_delay = df['Payment_Delay_Days'].mean()
avg_consistency = df['Payment_Consistency_Index'].mean()
avg_response = df['Response_to_Reminder_Ratio'].mean()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='{style_card(total_outstanding, [1_000_000, 500_000], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                f"💰 <b>Total Invoice:</b> ₹{total_invoice:,.0f}<br>"
                f"📌 <b>Total Outstanding:</b> ₹{total_outstanding:,.0f}</div>", unsafe_allow_html=True)

with col2:
    st.markdown(f"<div style='{style_card(avg_delay, [30, 10], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                f"⏱️ <b>Avg Payment Delay:</b> {avg_delay:.2f} days</div>", unsafe_allow_html=True)

with col3:
    st.markdown(f"<div style='{style_card(avg_consistency, [0.5, 0.8], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                f"📈 <b>Avg Consistency Index:</b> {avg_consistency:.2f}<br>"
                f"📬 <b>Avg Response Ratio:</b> {avg_response:.2f}</div>", unsafe_allow_html=True)

st.markdown("---")

# ==============================
# CUSTOMER SUMMARY
# ==============================
st.subheader("📊 Customer Segmentation & Risk Overview")
customer_summary = get_customer_summary(df)
st.dataframe(customer_summary)

# ==============================
# SEGMENTATION GRAPH
# ==============================
st.subheader("👥 Customer Segmentation Visualization")
fig = px.scatter(
    customer_summary,
    x="Invoice_Amount",
    y="Outstanding_Amount",
    size="Outstanding_Amount",
    color="ML_Risk",
    hover_name="Customer_ID",
    title="Customer Segmentation: Invoice vs Outstanding",
    labels={"Invoice_Amount": "Total Invoice", "Outstanding_Amount": "Total Outstanding"}
)
st.plotly_chart(fig, use_container_width=True)

# ==============================
# INDIVIDUAL CUSTOMER VIEW
# ==============================
st.subheader("🔍 Individual Customer View")
selected_customer = st.selectbox("Select Customer ID", customer_summary["Customer_ID"].unique())

if selected_customer:
    cust_data = df[df["Customer_ID"] == selected_customer]
    cust_summary = customer_summary[customer_summary["Customer_ID"] == selected_customer]

    st.write("### 🎯 Customer Metrics (Stylish Dashboard)")

    # Extract values
    total_invoice = cust_summary['Invoice_Amount'].values[0]
    outstanding = cust_summary['Outstanding_Amount'].values[0]
    delay = cust_summary['Payment_Delay_Days'].values[0]
    consistency = cust_summary['Payment_Consistency_Index'].values[0]
    response = cust_summary['Response_to_Reminder_Ratio'].values[0]
    rule_risk = cust_summary['Rule_Based_Risk'].values[0]
    ml_risk = cust_summary['ML_Risk'].values[0]

    # KPI Cards in Grid
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div style='{style_card(outstanding, [100000, 50000], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                    f"💰 <b>Total Invoice:</b> ₹{total_invoice:,.0f}<br>"
                    f"📌 <b>Outstanding:</b> ₹{outstanding:,.0f}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(f"<div style='{style_card(delay, [30, 10], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                    f"⏱️ <b>Avg Payment Delay:</b> {delay:.2f} days</div>", unsafe_allow_html=True)

    with col3:
        st.markdown(f"<div style='{style_card(consistency, [0.5, 0.8], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                    f"📈 <b>Consistency Index:</b> {consistency:.2f}</div>", unsafe_allow_html=True)

    col4, col5 = st.columns(2)
    with col4:
        st.markdown(f"<div style='{style_card(response, [0.5, 0.8], ['#ff4c4c','#FFD700','#90EE90'])}'>"
                    f"📬 <b>Reminder Response:</b> {response:.2f}</div>", unsafe_allow_html=True)

    with col5:
        st.markdown(f"<div style='background-color:#D3D3D3; padding:20px; border-radius:12px; text-align:center; font-size:18px;'>"
                    f"⚖️ <b>Rule-Based Risk:</b> {rule_risk}<br>"
                    f"🤖 <b>ML Risk:</b> {ml_risk}</div>", unsafe_allow_html=True)

    st.write("### 📑 Customer Invoices")
    st.dataframe(cust_data)

# ==============================
# EXPORT PDF
# ==============================
st.subheader("📤 Export Reports")
pdf_file = generate_pdf(customer_summary)
st.download_button("📄 Download PDF Report", data=pdf_file, file_name="customer_summary.pdf")
