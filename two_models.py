import pandas as pd
import numpy as np
import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# Function to connect to BigQuery and fetch data
@st.cache_data
def load_data():
    """Loads data from a BigQuery table using credentials from environment or Streamlit secrets."""
    try:
        if st.secrets.get("gcp_service_account"):
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            client = bigquery.Client(credentials=credentials)
        else:
            st.warning("No secrets found. Attempting to authenticate locally.")
            client = bigquery.Client()

        query = """
            SELECT * FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
        """
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error fetching data from BigQuery: {e}")
        st.warning("Please check your credentials and BigQuery table. Using local CSV for demonstration.")
        return pd.read_csv('synthetic_ar_dataset_noisy.csv')

# Function for data preprocessing and feature engineering
def preprocess_data(df):
    """Cleans data and creates new features for the model."""
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])
    df['Due_Date'] = pd.to_datetime(df['Due_Date'])
    df['Outstanding_Amount'] = df['Invoice_Amount'] - df['Amount_Paid']
    df['Payment_to_Invoice_Ratio'] = df['Amount_Paid'] / df['Invoice_Amount']
    
    df['Last_Payment_Date'] = pd.to_datetime(df['Last_Payment_Date'])
    current_date = pd.Timestamp.now().normalize()
    df['Days_Past_Due'] = (current_date - df['Due_Date']).dt.days
    
    features = ['Invoice_Amount', 'Outstanding_Amount', 'Payment_Delay_Days',
                'Partial_Payment_Flag', 'Payment_Consistency_Index',
                'Credit_Utilization_Velocity', 'Negotiation_Frequency',
                'Response_to_Reminder_Ratio', 'Days_Past_Due', 'Payment_to_Invoice_Ratio']
    
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    df_processed = df.dropna(subset=features)
    
    if 'High_Risk_Flag' not in df_processed.columns:
        st.error("The 'High_Risk_Flag' column is missing from the dataset. Cannot train model.")
        return None, None
        
    return df_processed, features

# Function to train and save the model
def train_model(df, features):
    """Trains a Random Forest Classifier and returns the model."""
    X = df[features]
    y = df['High_Risk_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"**F1 Score:** {f1:.2f}")
    st.sidebar.write(f"**Precision:** {precision:.2f}")
    st.sidebar.write(f"**Recall:** {recall:.2f}")
    
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.sidebar.subheader("Feature Importance")
    st.sidebar.bar_chart(feature_importances)
    
    return model

# Main application logic
def main():
    st.set_page_config(layout="wide", page_title="AR Predictive Collections Dashboard")
    st.title("Accounts Receivable Predictive Dashboard 🔮")
    st.markdown("A proactive tool to predict and prioritize high-risk invoices, powered by machine learning.")

    df, features = preprocess_data(load_data())
    
    if df is None or df.empty or features is None:
        st.warning("Data could not be loaded or is empty after cleaning.")
        return
    
    st.info("Training the predictive model... This may take a moment.")
    model = train_model(df, features)
    
    df['risk_probability'] = model.predict_proba(df[features])[:, 1]
    
    # --- Start of New Clustering Logic ---
    # Define clustering logic
    def get_risk_cluster(prob):
        if prob >= 0.7:
            return 'High Risk'
        elif 0.4 <= prob < 0.7:
            return 'Medium Risk'
        else:
            return 'Low Risk'

    df['Risk_Cluster'] = df['risk_probability'].apply(get_risk_cluster)
    
    # Define colors for clustering
    cluster_colors = {
        'High Risk': 'red',
        'Medium Risk': 'yellow',
        'Low Risk': 'green'
    }
    
    # Map cluster names to hex colors for the scatter chart
    df['Cluster_Color'] = df['Risk_Cluster'].map(cluster_colors)
    # --- End of New Clustering Logic ---
    
    st.sidebar.header("Filter by Customer Industry")
    industries = ['All'] + sorted(df['Customer_Industry'].unique().tolist())
    selected_industry = st.sidebar.selectbox("Select an industry:", industries)
    
    if selected_industry != 'All':
        df_display = df[df['Customer_Industry'] == selected_industry]
    else:
        df_display = df.copy()

    df_display = df_display.sort_values(by='risk_probability', ascending=False)

    st.header("Key Performance Indicators 📈")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_outstanding = df_display['Outstanding_Amount'].sum()
        st.metric(label="Total Outstanding 💰", value=f"${total_outstanding:,.2f}")
    
    with col2:
        high_risk_invoices = df_display[df_display['risk_probability'] >= 0.5].shape[0]
        st.metric(label="High-Risk Invoices ⚠️", value=f"{high_risk_invoices}")
    
    with col3:
        avg_payment_delay = df_display['Payment_Delay_Days'].mean()
        st.metric(label="Avg. Payment Delay (Days) ⏳", value=f"{avg_payment_delay:.2f}")
    
    with col4:
        avg_invoice_amount = df_display['Invoice_Amount'].mean()
        st.metric(label="Avg. Invoice Amount 🧾", value=f"${avg_invoice_amount:,.2f}")

    with col5:
        avg_risk_score = df_display['risk_probability'].mean()
        st.metric(label="Avg. Risk Score ⭐", value=f"{avg_risk_score:.2%}")

    st.header("Visual Insights 📊")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Outstanding Amount by Industry")
        industry_summary = df_display.groupby('Customer_Industry')['Outstanding_Amount'].sum().sort_values(ascending=False)
        st.bar_chart(industry_summary)
        
    with col_b:
        st.subheader("Invoice Amount vs. Risk Probability")
        # Use a dictionary to set colors for the scatter chart
        st.scatter_chart(df_display, x='Invoice_Amount', y='risk_probability', color='Cluster_Color')

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Distribution of Payment Delays")
        st.hist_chart(df_display, x='Payment_Delay_Days', bins=50)

    with col_d:
        st.subheader("Count of Customers by Risk Cluster")
        cluster_counts = df_display['Risk_Cluster'].value_counts()
        st.bar_chart(cluster_counts, color=['#FF0000', '#FFFF00', '#00FF00']) # Using hex codes for color

    st.subheader("Invoices Sorted by Risk Probability (Highest First)")
    
    display_cols = ['Invoice_No', 'Customer_ID', 'Customer_Industry', 'Outstanding_Amount', 
                    'Status', 'Days_Past_Due', 'risk_probability', 'Risk_Cluster']
    
    df_display['Outstanding_Amount'] = df_display['Outstanding_Amount'].apply(lambda x: f'${x:,.2f}')
    df_display['risk_probability'] = df_display['risk_probability'].apply(lambda x: f'{x:.2%}')
    
    st.dataframe(df_display[display_cols].rename(columns={
        'risk_probability': 'Predicted Risk Probability',
        'Outstanding_Amount': 'Outstanding Amount',
        'Risk_Cluster': 'Customer Risk Cluster'
    }))

if __name__ == "__main__":
    main()
