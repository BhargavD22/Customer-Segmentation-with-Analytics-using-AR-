import pandas as pd
import streamlit as st
import joblib
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# Function to connect to BigQuery and fetch data
@st.cache_data
def load_data():
    """Loads data from a BigQuery table."""
    try:
        # NOTE: This assumes you have set up your Google Cloud credentials
        client = bigquery.Client(project='mss-data-engineer-sandbox')
        query = """
            SELECT * FROM `mss-data-engineer-sandbox.customer_segmentation_using_AR.csusingar`
        """
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        st.error(f"Error fetching data from BigQuery: {e}")
        st.warning("Please ensure your credentials are set up and the table exists. Using local CSV for demonstration.")
        # Fallback to local CSV for demonstration if BQ connection fails
        return pd.read_csv('synthetic_ar_dataset_noisy.csv')

# Function for data preprocessing and feature engineering
def preprocess_data(df):
    """Cleans data and creates new features for the model."""
    df['Invoice_Date'] = pd.to_datetime(df['Invoice_Date'])
    df['Due_Date'] = pd.to_datetime(df['Due_Date'])
    df['Outstanding_Amount'] = df['Invoice_Amount'] - df['Amount_Paid']
    df['Payment_to_Invoice_Ratio'] = df['Amount_Paid'] / df['Invoice_Amount']
    
    # Fill missing dates for overdue invoices
    df['Last_Payment_Date'] = pd.to_datetime(df['Last_Payment_Date'])
    current_date = pd.Timestamp.now().normalize()
    df['Days_Past_Due'] = (current_date - df['Due_Date']).dt.days
    
    # Handle NaNs in numerical columns
    df['Payment_to_Invoice_Ratio'] = df['Payment_to_Invoice_Ratio'].fillna(0)
    
    # Select features for the model
    features = ['Invoice_Amount', 'Outstanding_Amount', 'Payment_Delay_Days',
                'Partial_Payment_Flag', 'Payment_Consistency_Index',
                'Credit_Utilization_Velocity', 'Negotiation_Frequency',
                'Response_to_Reminder_Ratio', 'Days_Past_Due', 'Payment_to_Invoice_Ratio']
    
    # Handle potential NaNs in features
    df = df.dropna(subset=['Payment_Delay_Days'])
    
    return df, features

# Function to train and save the model
def train_model(df, features):
    """Trains a Random Forest Classifier and returns the model."""
    X = df[features]
    y = df['High_Risk_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"**F1 Score:** {f1:.2f}")
    st.sidebar.write(f"**Precision:** {precision:.2f}")
    st.sidebar.write(f"**Recall:** {recall:.2f}")
    
    # Get feature importances and display in the sidebar
    feature_importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    st.sidebar.subheader("Feature Importance")
    st.sidebar.bar_chart(feature_importances)
    
    return model

# Main application logic
def main():
    st.set_page_config(layout="wide", page_title="AR Predictive Collections Dashboard")
    st.title("Accounts Receivable Predictive Dashboard 🔮")

    # Load and preprocess data
    df = load_data()
    if df is None or df.empty:
        st.warning("Data could not be loaded. Please check your credentials and table name.")
        return

    df, features = preprocess_data(df)
    
    st.info("Training the predictive model... This may take a moment.")
    model = train_model(df, features)
    
    # Generate predictions
    df['risk_probability'] = model.predict_proba(df[features])[:, 1]
    
    # Create sidebar for filtering
    st.sidebar.header("Filter by Customer Industry")
    industries = ['All'] + sorted(df['Customer_Industry'].unique().tolist())
    selected_industry = st.sidebar.selectbox("Select an industry:", industries)
    
    if selected_industry != 'All':
        df_display = df[df['Customer_Industry'] == selected_industry]
    else:
        df_display = df.copy()

    # Sort invoices by risk score
    df_display = df_display.sort_values(by='risk_probability', ascending=False)
    
    # KPI metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_outstanding = df_display['Outstanding_Amount'].sum()
        st.metric(label="Total Outstanding Amount 💰", value=f"${total_outstanding:,.2f}")
    
    with col2:
        high_risk_invoices = df_display[df_display['risk_probability'] >= 0.5].shape[0]
        total_invoices = df_display.shape[0]
        st.metric(label="High-Risk Invoices ⚠️", value=f"{high_risk_invoices} of {total_invoices}")
    
    with col3:
        avg_payment_delay = df_display['Payment_Delay_Days'].mean()
        st.metric(label="Avg. Payment Delay (Days) ⏳", value=f"{avg_payment_delay:.2f}")
    
    # Main table display
    st.subheader("Invoices Sorted by Risk Probability (Highest First)")
    
    # Display the most relevant columns
    display_cols = ['Invoice_No', 'Customer_ID', 'Customer_Industry', 'Outstanding_Amount', 
                    'Status', 'Days_Past_Due', 'risk_probability']
    
    # Format the table for better readability
    df_display['Outstanding_Amount'] = df_display['Outstanding_Amount'].apply(lambda x: f'${x:,.2f}')
    df_display['risk_probability'] = df_display['risk_probability'].apply(lambda x: f'{x:.2%}')
    
    st.dataframe(df_display[display_cols].rename(columns={
        'risk_probability': 'Predicted Risk Probability',
        'Outstanding_Amount': 'Outstanding Amount'
    }))

if __name__ == "__main__":
    main()
