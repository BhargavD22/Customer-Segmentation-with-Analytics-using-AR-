import streamlit as st
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import plotly.express as px
from google.api_core.exceptions import GoogleAPIError

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Customer AR Insights Dashboard", layout="wide")

# ==============================
# STYLING
# ==============================
MIRACLE_BLUE = "#00AEEF"
MIRACLE_RED = "#FF4B4B"
MIRACLE_GREEN = "#2ECC71"

st.markdown(
    """
    <style>
    .kpi-card {
        background-color: white;
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
    }
    .kpi-card h4 {
        margin: 0;
        color: #333;
        font-weight: 600;
    }
    .kpi-card p {
        margin: 5px 0;
        font-size: 22px;
        font-weight: bold;
    }
    .logo-container {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 20px;
    }
    .logo-container h1 {
        color: #333;
        margin: 0;
    }
    .footer {
        margin-top: 50px;
        margin-bottom: 10px;
        text-align: center;
        opacity: 0.7;
    }
    .footer img {
        width: 140px;
    }
    .footer p {
        margin-top: 5px;
        font-size: 13px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# A more robust way to handle the logo and title HTML
LOGO_URL = "https://i.imgur.com/your-public-logo-url.png" # Replace with a public URL for deployment
TITLE_HTML = f"""
    <div class="logo-container">
        <img src="{LOGO_URL}" alt="Miracle Logo" width="220">
        <h1>Customer AR Insights Dashboard</h1>
    </div>
"""
st.markdown(TITLE_HTML, unsafe_allow_html=True)
st.write("Data is securely fetched from **BigQuery** or uploaded via **CSV**.")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data_from_bigquery():
    try:
        # NOTE: This temporary fix hardcodes the credentials to bypass a Streamlit secrets issue.
        # In a production environment, you should use st.secrets.
        gcp_service_account_dict = {
            "type": "service_account",
            "project_id": "mss-data-engineer-sandbox",
            "private_key_id": "f301697db532c16b549026a2de0fa650c5e8a8e4",
            "private_key": """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDEBAQKGaJ4s95C
s/E/OvUSjVD5USB0k9jC7MHVxidEGDp0ccO5B8mB3/pjrVF80LiuBz3RAaYDHaN9
AOVlcCp/QNU0R84hTFxp/6TiywNF6aPjjzcD9KJt4ALFVl9aVT5hre/V11utjGag
tUnf6n6FPA4+lbPKwm17TRaf+qyTENZtlZxgU2lRVpe7oO4wl4EkqLkkI5ltbA8x
iFaixx9hWTWkVgzUKaZCFZb6cZjaKBVoFNE2Le5PZN0CDuaHgj58NoJy4P9aCv6q
UFPfGxBnHx9B27DOKTHzKthiIYGCEQkkhYfIDhc9IA9D1ftIl0iRGuYNy11+BQFt
D30He6UfAgMBAAECggEAMBeqJA5BEitTc3sxzCk7eudIQDt64o9pxf2P21LoTGlD
YlGNS2cRNjfNd8pM7XpKbYxiStcEM8yAUcm3/sjj7F/sT4z0kq/pFqq9+lUEAxG9
f7YiMerCNYIaO++iqoeyrAWgjA9wM2b4wSJpszIWA7uF5S4WtD863F9AF1VKJTS0
uBcPI8NzkqJTyYWFtmSIObRZpRH81VTm4S3knS4WsNtN77KjTOZkA6+MQdARq1KA
9MWRvU/BOSsPrzTUV1GyLItuw8MefJnX0SPrLTx5SYX5ePNkeJe1qrxCWPeTZfmN
WXUviUEDbGh+0VX6Do4skQ0/HU6RDo+19Yir7HIYtQKBgQDvB+ta1cAyhMKWqhBH
Qu7HUFpDYCWt9/aMIJ6GK49XMq0oyeAnwj6vwfzcknQSmph6pHUPrSOchc0pJiAT
ueQLj1UfwDo+XrDT5pMUXQOo07PYfVxM2jX790sqewbBxMTgUMaQK8nY/yrHZ4e7
e90aEw65pHKT9FNqbIouexMiawKBgQDR7ll2N+mOQFEQ54NWpgsC+2gwyiKec8ys
7cN88Yr4fsEVwtUubtAtZwEYMMFsXt/lquYDxKPoZHNZfOSV50Yd38KIrI43PR29
mGYZmyP7UmDG1BYZTt0JcjDoCMdFGZmDkW9UvZqorrH8yc0HbF7PRhzK0Mo5zVA9
wOoojSf9HQKBgQDfHx5TaQmCXqihKNgPHOx0wo2vLLWfYcITZXN0PH8N3zEBzQdf
NZN8TnDxmAefQg2pFZBr9Ks0NTWf/oWcxD2ZiM7l13LGu28GLcoHDRgYZJ0RLVuW
JW6U526Tlcll4H4CAYSIGUfONcnB3uM1X9awuy9YnKeTclLXGcAWyS3ARwKBgCjZ
B+9I0dksCpoPci7aACqEYLGdoz7RqXG8kd0t4qyXfVqOnox6Y2dyM3RRiFFd5JL7
veXdzUbaxNcUxiWk6q/FakTNzp5Q9gh+Lt+soEO2s738ZpBmF/xOi9WaX6vCX2yK
T+9dNUq9M0TMv2hCXfBW5CNSnQbCPGrHrshVLwLBAoGBAOjmWD22u53wQNwpVlbo
F3wjZ+eXnaqB32KG9fhuP1NJfVaZ65INBLboBgfTbsNUXqD+tk5pgDPWF+APNMa/
1ZuHqqUsY7ybsX/LVBYlt/gVJmFFRWFy/dOqLJIQ9iu+KSldYlSBdVYc4kkLMYPe
IO3duHMlIwG8BOBMW9ASoGvE
-----END PRIVATE KEY-----""",
            "client_email": "dataengg-training@mss-data-engineer-sandbox.iam.gserviceaccount.com",
            "client_id": "116876153342903559080",
            "token_uri": "https://oauth2.googleapis.com/token",
            "type": "service_account"
        }
        credentials = service_account.Credentials.from_service_account_info(gcp_service_account_dict)
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        
        query = """
            SELECT *
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
    uploaded_file = st.file_uploader("Upload your CSV", type="csv")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            df = None
    else:
        st.info("Please upload a CSV file to continue.")

if df is None or df.empty:
    st.stop()

# ==============================
# KPI METRICS (Minimalist Style)
# ==============================
st.markdown("### 📌 Key Business KPIs")

def kpi_card(title, value, unit="", threshold=None, higher_is_bad=True):
    if threshold is not None:
        if higher_is_bad:
            color = MIRACLE_RED if value > threshold else MIRACLE_GREEN
        else:
            color = MIRACLE_GREEN if value > threshold else MIRACLE_RED
    else:
        color = MIRACLE_BLUE

    return f"""
        <div class="kpi-card">
            <h4>{title}</h4>
            <p style="color:{color};">{value:,.2f}{unit}</p>
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
fig.update_traces(line=dict(color=MIRACLE_BLUE))
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
    f"""
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div class="footer">
        <img src="{LOGO_URL}" alt="Miracle Logo">
        <p>© 2025 Miracle Software Systems - All Rights Reserved</p>
    </div>
    """,
    unsafe_allow_html=True,
)
