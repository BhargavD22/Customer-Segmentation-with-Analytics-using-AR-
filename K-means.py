# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
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
# SESSION STATE INIT
# ==============================
if "data_source" not in st.session_state:
    st.session_state["data_source"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None

# ==============================
# DATA LOADING
# ==============================
def load_data_from_bigquery():
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["bigquery"]
    )
    client = bigquery.Client(credentials=credentials, project=credentials.project_id)
    query = """
        SELECT *
        FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
    """
    return client.query(query).to_dataframe()

def load_data_from_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

# ==============================
# PDF GENERATION (UNICODE + EMOJIS)
# ==============================
def generate_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    
    # Load bundled Unicode font (make sure fonts/DejaVuSans.ttf exists in repo)
    font_path = os.path.join("fonts", "DejaVuSans.ttf")
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 14)
    
    pdf.cell(200, 10, txt="📊 Customer AR Summary Report", ln=True, align='C')
    pdf.set_font("DejaVu", "", 12)

    for _, row in data.iterrows():
        pdf.ln(5)
        pdf.cell(200, 10,
                 txt=f"🧑 Customer: {row['Customer_ID']} | 🟢 Rule Risk: {row['Rule_Based_Risk']} | 🔴 ML Risk: {row['ML_Risk']}",
                 ln=True)
        pdf.cell(200, 10,
                 txt=f"💰 Total Invoice: ₹{row['Invoice_Amount']:,.0f}, 📌 Outstanding: ₹{row['Outstanding_Amount']:,.0f}",
                 ln=True)

    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

# ==============================
# DATA SOURCE SELECTION (ONCE)
# ==============================
if st.session_state["data_source"] is None:
    st.info("Please select a data source to continue.")
    option = st.radio("Select Data Source", ["BigQuery", "CSV Upload"])

    if option == "BigQuery":
        with st.spinner("Fetching data from BigQuery..."):
            df = load_data_from_bigquery()
        st.session_state["df"] = df
        st.session_state["data_source"] = "BigQuery"
        st.success("✅ Data loaded from BigQuery")

    elif option == "CSV Upload":
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded_file is not None:
            df = load_data_from_csv(uploaded_file)
            st.session_state["df"] = df
            st.session_state["data_source"] = "CSV"
            st.success("✅ Data loaded from CSV")

# ==============================
# MAIN APP (AFTER DATA IS LOADED)
# ==============================
if st.session_state["df"] is not None:
    df = st.session_state["df"]

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # K-Means clustering
    st.subheader("📊 K-Means Clustering")
    features = ["Invoice_Amount", "Outstanding_Amount", "Payment_Delay_Days"]
    df_cluster = df[features].fillna(0)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(df_cluster)

    fig = px.scatter(
        df, x="Invoice_Amount", y="Outstanding_Amount",
        color="Cluster", size="Payment_Delay_Days",
        hover_data=["Customer_ID"]
    )
    st.plotly_chart(fig, use_container_width=True)

    # Customer filter
    st.subheader("🔎 Customer Details")
    customer_id = st.selectbox("Select Customer", df["Customer_ID"].unique())
    customer_summary = df[df["Customer_ID"] == customer_id]

    st.write("### 📑 Customer Summary")
    st.table(customer_summary)

    # PDF download
    pdf_bytes = generate_pdf(customer_summary)
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"customer_{customer_id}_report.pdf",
        mime="application/pdf",
    )
