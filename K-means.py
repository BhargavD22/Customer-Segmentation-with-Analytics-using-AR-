# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from google.cloud import bigquery
from google.oauth2 import service_account

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(layout="wide", page_title="AR Analytics & Customer Segmentation")

st.title("📊 Customer AR Insights Dashboard")
st.markdown("Data is securely fetched from **BigQuery** or uploaded via **CSV**.")

# ==============================
# SESSION STATE INITIALIZATION
# ==============================
if "df" not in st.session_state:
    st.session_state.df = None
if "data_source" not in st.session_state:
    st.session_state.data_source = None

# ==============================
# CREDENTIALS (from .streamlit/secrets.toml)
# ==============================
if "bigquery" in st.secrets:
    credentials = service_account.Credentials.from_service_account_info(st.secrets["bigquery"])
    project_id = st.secrets["bigquery"]["project_id"]
else:
    credentials = None
    project_id = None

# ==============================
# DATA SOURCE SELECTION
# ==============================
st.subheader("Select Data Source:")
data_source = st.radio("Choose source:", ["BigQuery", "Upload CSV"], index=0)

st.session_state.data_source = data_source

if data_source == "BigQuery":
    if st.button("🔗 Fetch Data from BigQuery"):
        try:
            client = bigquery.Client(credentials=credentials, project=project_id)
            query = """
                SELECT *
                FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
            """
            df = client.query(query).to_dataframe()

            # Parse date fields
            df["Invoice_Date"] = pd.to_datetime(df["Invoice_Date"], errors="coerce")
            df["Due_Date"] = pd.to_datetime(df["Due_Date"], errors="coerce")
            df["Last_Payment_Date"] = pd.to_datetime(df["Last_Payment_Date"], errors="coerce")

            st.session_state.df = df
            st.success("✅ Data successfully fetched from BigQuery!")

        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your AR CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, parse_dates=["Invoice_Date", "Due_Date", "Last_Payment_Date"])
        st.session_state.df = df
        st.success("✅ File successfully uploaded and parsed.")

# ==============================
# MAIN DASHBOARD (only if df exists)
# ==============================
if st.session_state.df is not None:
    df = st.session_state.df

    # BASIC CLEANUP
    df["Payment_Delay_Days"] = pd.to_numeric(df["Payment_Delay_Days"], errors="coerce")
    df["Invoice_Amount"] = pd.to_numeric(df["Invoice_Amount"], errors="coerce")
    df["Outstanding_Amount"] = pd.to_numeric(df["Outstanding_Amount"], errors="coerce")

    # ==============================
    # KPI SECTION
    # ==============================
    st.subheader("📌 Key Business KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("💰 Total Invoice Amount", f"₹ {df['Invoice_Amount'].sum():,.0f}")
    col2.metric("📉 Total Outstanding", f"₹ {df['Outstanding_Amount'].sum():,.0f}")
    col3.metric("⏱️ Avg Payment Delay", f"{df['Payment_Delay_Days'].mean():.2f} days")
    col4.metric("📈 Avg Consistency Index", f"{df['Payment_Consistency_Index'].mean():.2f}")
    col5.metric("📬 Avg Response Ratio", f"{df['Response_to_Reminder_Ratio'].mean():.2f}")

    st.markdown("---")

    # ==============================
    # CUSTOMER SUMMARY
    # ==============================
    st.subheader("📊 Customer Segmentation & Risk Overview")

    customer_summary = df.groupby("Customer_ID").agg({
        "Invoice_Amount": "sum",
        "Outstanding_Amount": "sum",
        "Payment_Delay_Days": "mean",
        "Payment_Consistency_Index": "mean",
        "Response_to_Reminder_Ratio": "mean",
        "High_Risk_Flag": "max"
    }).reset_index()

    # RULE-BASED RISK
    customer_summary["Rule_Based_Risk"] = customer_summary["High_Risk_Flag"].apply(
        lambda x: "🔴 High" if x == 1 else "🟢 Low"
    )

    # ==============================
    # ML CLUSTERING (K-Means)
    # ==============================
    features = customer_summary[[
        "Invoice_Amount", "Outstanding_Amount", "Payment_Delay_Days",
        "Payment_Consistency_Index", "Response_to_Reminder_Ratio"
    ]].fillna(0)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    customer_summary["Cluster"] = kmeans.fit_predict(scaled_features)

    risk_labels = {0: "🟢 Low", 1: "🟡 Medium", 2: "🔴 High"}
    customer_summary["ML_Risk"] = customer_summary["Cluster"].map(risk_labels)

    st.dataframe(customer_summary)

    # ==============================
    # TRENDS
    # ==============================
    st.subheader("📈 Invoice & Outstanding Trend")

    df["Invoice_Month"] = df["Invoice_Date"].dt.to_period("M").astype(str)
    monthly_summary = df.groupby("Invoice_Month")[["Invoice_Amount", "Outstanding_Amount"]].sum().reset_index()

    fig = px.line(
        monthly_summary, x="Invoice_Month", y=["Invoice_Amount", "Outstanding_Amount"],
        labels={"value": "Amount", "variable": "Metric"}, markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==============================
    # CUSTOMER COMPARISON
    # ==============================
    st.subheader("👥 Compare Customers")

    fig2 = px.scatter(
        customer_summary, x="Payment_Delay_Days", y="Outstanding_Amount",
        color="ML_Risk", size="Invoice_Amount", hover_name="Customer_ID",
        title="Outstanding vs Delay with ML Risk Clusters"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ==============================
    # INDIVIDUAL CUSTOMER VIEW
    # ==============================
    st.subheader("🔍 Individual Customer View")
    selected_customer = st.selectbox("Select a Customer ID", customer_summary["Customer_ID"].unique())

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

    # ==============================
    # EXPORT SECTION
    # ==============================
    st.subheader("📤 Export Reports")

    def to_excel(df):
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name="Customer_Summary")
        return out.getvalue()

    excel = to_excel(customer_summary)
    st.download_button("📥 Download Excel Report", data=excel, file_name="customer_summary.xlsx")

    def generate_pdf(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="Customer AR Summary Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        for i, row in data.iterrows():
            pdf.ln(5)
            pdf.cell(200, 10, txt=f"Customer: {row['Customer_ID']} | Rule Risk: {row['Rule_Based_Risk']} | ML Risk: {row['ML_Risk']}", ln=True)
            pdf.cell(200, 10, txt=f"Total Invoice: ₹{row['Invoice_Amount']:,.0f}, Outstanding: ₹{row['Outstanding_Amount']:,.0f}", ln=True)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        return pdf_output.getvalue()

    pdf_file = generate_pdf(customer_summary)
    st.download_button("📄 Download PDF Report", data=pdf_file, file_name="customer_summary.pdf")

else:
    st.info("ℹ️ Please load data from BigQuery or CSV to proceed.")
