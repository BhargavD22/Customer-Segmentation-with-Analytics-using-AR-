import streamlit as st
import pandas as pd
import plotly.express as px
from fpdf import FPDF
import io
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from google.cloud import bigquery
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPIError

st.set_page_config(layout="wide", page_title="AR Analytics & Customer Segmentation")

st.title("📊 Customer AR Insights Dashboard")
st.markdown("Data is securely fetched from **BigQuery** or uploaded via **CSV**.")

# ==============================
# LOAD DATA
# ==============================
def load_data_from_bigquery():
    try:
        credentials_info = st.secrets["gcp_service_account"]
        credentials = service_account.Credentials.from_service_account_info(
            credentials_info
        )
        client = bigquery.Client(credentials=credentials, project=credentials_info["project_id"])
        
        query = """
            SELECT Customer_ID, Invoice_Date, Due_Date, Last_Payment_Date, Payment_Delay_Days,
                   Invoice_Amount, Outstanding_Amount, Payment_Consistency_Index,
                   Response_to_Reminder_Ratio, High_Risk_Flag
            FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
        """
        df = client.query(query).to_dataframe()
        return df
    except GoogleAPIError as e:
        st.error(f"Error connecting to BigQuery: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# File uploader option
data_source = st.radio("Select Data Source:", ["BigQuery", "Upload CSV"], horizontal=True)

df = None
if data_source == "BigQuery":
    df = load_data_from_bigquery()
elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your AR CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["Invoice_Date", "Due_Date", "Last_Payment_Date"])
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            df = None
    else:
        st.info("Please upload a CSV file to continue.")
        st.stop()
        
if df is None or df.empty:
    st.stop()

# BASIC CLEANUP & FEATURE ENGINEERING
df["Payment_Delay_Days"] = pd.to_numeric(df["Payment_Delay_Days"], errors="coerce")
df["Invoice_Amount"] = pd.to_numeric(df["Invoice_Amount"], errors="coerce")
df["Outstanding_Amount"] = pd.to_numeric(df["Outstanding_Amount"], errors="coerce")
df.dropna(subset=["Payment_Delay_Days", "Outstanding_Amount"], inplace=True)

# ==============================
# DASHBOARD LAYOUT
# ==============================

st.subheader("📌 Key Business KPIs")
col1, col2, col3, col4 = st.columns(4)
col1.metric("💰 Total Invoice Amount", f"₹ {df['Invoice_Amount'].sum():,.0f}")
col2.metric("📉 Total Outstanding", f"₹ {df['Outstanding_Amount'].sum():,.0f}")
col3.metric("⏱️ Avg Payment Delay", f"{df['Payment_Delay_Days'].mean():.2f} days")
col4.metric("📊 Total Customers", f"{df['Customer_ID'].nunique()}")

st.markdown("---")

# K-MEANS SEGMENTATION
st.subheader("📊 Customer Segmentation using K-Means")

features = df.groupby("Customer_ID").agg(
    Outstanding_Amount=("Outstanding_Amount", "sum"),
    Payment_Delay_Days=("Payment_Delay_Days", "mean")
).reset_index()

num_clusters = st.slider("Select the number of customer segments (K)", min_value=2, max_value=10, value=3)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features[["Outstanding_Amount", "Payment_Delay_Days"]])

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
features["Cluster"] = kmeans.fit_predict(scaled_features)

customer_summary = df.groupby("Customer_ID").agg({
    "Invoice_Amount": "sum",
    "Outstanding_Amount": "sum",
    "Payment_Delay_Days": "mean",
    "Payment_Consistency_Index": "mean",
    "Response_to_Reminder_Ratio": "mean",
    "High_Risk_Flag": "max"
}).reset_index()
customer_summary = pd.merge(customer_summary, features[["Customer_ID", "Cluster"]], on="Customer_ID")
customer_summary["Risk_Score"] = customer_summary["High_Risk_Flag"].apply(lambda x: "🔴 High" if x == 1 else "🟢 Low")

st.dataframe(customer_summary)

# TREND VISUALS
st.subheader("📈 Invoice & Outstanding Trend")

df["Invoice_Month"] = pd.to_datetime(df["Invoice_Date"]).dt.to_period("M").astype(str)
monthly_summary = df.groupby("Invoice_Month")[["Invoice_Amount", "Outstanding_Amount"]].sum().reset_index()

fig = px.line(monthly_summary, x="Invoice_Month", y=["Invoice_Amount", "Outstanding_Amount"],
              labels={"value": "Amount", "variable": "Metric"}, markers=True)
st.plotly_chart(fig, use_container_width=True)

# CUSTOMER COMPARISON
st.subheader("👥 Compare Customers")

fig2 = px.scatter(customer_summary, x="Payment_Delay_Days", y="Outstanding_Amount",
                  color="Cluster", size="Invoice_Amount", hover_name="Customer_ID",
                  title="Customer Clusters based on Outstanding and Payment Delay")
st.plotly_chart(fig2, use_container_width=True)

# INDIVIDUAL CUSTOMER VIEW
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
    st.metric("K-Means Cluster", cust_summary["Cluster"].values[0])

    st.write("### Invoices")
    st.dataframe(cust_data)

# EXPORT SECTION
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
        pdf.cell(200, 10, txt=f"Customer: {row['Customer_ID']} | Cluster: {row['Cluster']}", ln=True)
        pdf.cell(200, 10, txt=f"Total Invoice: ₹{row['Invoice_Amount']:,.0f}, Outstanding: ₹{row['Outstanding_Amount']:,.0f}", ln=True)
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()

pdf_file = generate_pdf(customer_summary)
st.download_button("📄 Download PDF Report", data=pdf_file, file_name="customer_summary.pdf")
