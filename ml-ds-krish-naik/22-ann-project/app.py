import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Analytics Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling (glassmorphism, fonts, animated progress bars, etc.)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Title container styling */
    .header-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        color: white;
        margin-bottom: 25px;
        border-left: 5px solid #00c6ff;
    }
    .header-title {
        font-size: 32px;
        font-weight: 700;
        margin: 0 0 10px 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .header-subtitle {
        font-size: 16px;
        opacity: 0.9;
        margin: 0;
        font-weight: 300;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #00c6ff;
        margin-bottom: 15px;
    }
    
    /* Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Result section glow colors */
    .result-card-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.15);
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
    }
    .result-card-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.15);
        border-radius: 16px;
        padding: 25px;
        margin-top: 15px;
    }
    
    /* Gauge styles */
    .gauge-container {
        background-color: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        height: 14px;
        width: 100%;
        margin: 15px 0;
        overflow: hidden;
        position: relative;
    }
    .gauge-fill {
        height: 100%;
        border-radius: 20px;
        transition: width 1s ease-in-out;
    }
    .gauge-low {
        background: linear-gradient(90deg, #10b981 0%, #34d399 100%);
    }
    .gauge-high {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 100%);
    }
    
    /* Metrics display */
    .metric-value {
        font-size: 42px;
        font-weight: 800;
        line-height: 1;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load cached models and encoders
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("./models/churn_model.h5")
    
    with open("./models/gender_label_encoder.pkl", "rb") as f:
        gender_encoder = pickle.load(f)
        
    with open("./models/one_hot_geography_encoder.pkl", "rb") as f:
        geo_encoder = pickle.load(f)
        
    with open("./models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    return model, gender_encoder, geo_encoder, scaler

# Load cached dataset
@st.cache_data
def load_dataset():
    df = pd.read_csv("./data/Churn_Modelling.csv")
    return df

# Initialize assets
try:
    model, gender_encoder, geo_encoder, scaler = load_assets()
    churn_df = load_dataset()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading models or dataset files: {e}")
    st.info("Please make sure your model files (models/churn_model.h5, models/scaler.pkl, etc.) are present in the models directory.")

# App Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🏦 Bank Customer Churn Analytics Portal</div>
    <div class="header-subtitle">Evaluate and predict customer attrition risk using our custom Deep Learning Artificial Neural Network (ANN) model.</div>
</div>
""", unsafe_allow_html=True)

if assets_loaded:
    # Sidebar contents
    st.sidebar.markdown('<div class="sidebar-title">⚡ ANN Model Configuration</div>', unsafe_allow_html=True)
    st.sidebar.write("The underlying classifier is an Artificial Neural Network (ANN) trained on historical bank customer records to identify churn patterns.")
    
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('**Model Details:**')
    st.sidebar.write("- **Framework:** TensorFlow / Keras")
    st.sidebar.write("- **Input Features:** 11 Dimensions")
    st.sidebar.write("- **Preprocessors:** Standard Scaler, One-Hot Encoder, Label Encoder")
    
    # Preset Customer profiles loading feature
    st.sidebar.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown('**📂 Quick Load Customer Profile**')
    st.sidebar.write("Select a real customer profile from the dataset to pre-populate the inputs below:")
    
    # Generate list of customer names for dropdown
    # Let's get a mix of stayed and exited customer profiles
    sample_customers = pd.concat([
        churn_df[churn_df['Exited'] == 1].head(10),
        churn_df[churn_df['Exited'] == 0].head(10)
    ]).sort_values(by='Surname')
    
    # Dropdown label format
    customer_options = ["-- Choose a Sample Customer --"] + [
        f"{row['Surname']} (Exited: {'Yes' if row['Exited'] == 1 else 'No'}, CS: {row['CreditScore']})"
        for idx, row in sample_customers.iterrows()
    ]
    
    selected_customer_label = st.sidebar.selectbox(
        "Select Profile",
        options=customer_options,
        index=0,
        label_visibility="collapsed"
    )
    
    # State tracking: default values dictionary
    defaults = {
        "credit_score": 650,
        "geography": "France",
        "gender": "Male",
        "age": 38,
        "tenure": 5,
        "balance": 0.0,
        "num_of_products": 2,
        "has_cr_card": 1,
        "is_active_member": 1,
        "estimated_salary": 50000.0
    }
    
    # Initialize session state variables if they are not already set
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
            
    # Handle sample customer loading
    if selected_customer_label != "-- Choose a Sample Customer --":
        # Extract Surname from the selected label
        surname = selected_customer_label.split(" (")[0]
        # Get corresponding row
        row = sample_customers[sample_customers['Surname'] == surname].iloc[0]
        
        # Update session state values
        st.session_state.credit_score = int(row['CreditScore'])
        st.session_state.geography = str(row['Geography'])
        st.session_state.gender = str(row['Gender'])
        st.session_state.age = int(row['Age'])
        st.session_state.tenure = int(row['Tenure'])
        st.session_state.balance = float(row['Balance'])
        st.session_state.num_of_products = int(row['NumOfProducts'])
        st.session_state.has_cr_card = int(row['HasCrCard'])
        st.session_state.is_active_member = int(row['IsActiveMember'])
        st.session_state.estimated_salary = float(row['EstimatedSalary'])
        
        # Clear sidebar selectbox value and rerun to apply changes
        st.sidebar.success(f"Loaded profile: {surname}")
        # Programmatic rerun to update fields
        st.rerun()

    # Create the Form inputs
    st.markdown('<div class="glass-card"><h3>📋 Customer Attributes Form</h3>', unsafe_allow_html=True)
    
    # Arrange fields in 3 columns for premium, spatial layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Demographics")
        geography = st.selectbox(
            "Geography (Country)",
            options=["France", "Germany", "Spain"],
            key="geography"
        )
        gender = st.selectbox(
            "Gender",
            options=["Male", "Female"],
            key="gender"
        )
        age = st.slider(
            "Age",
            min_value=18,
            max_value=100,
            key="age"
        )
        
    with col2:
        st.subheader("Financial Profile")
        credit_score = st.slider(
            "Credit Score",
            min_value=350,
            max_value=850,
            key="credit_score"
        )
        balance = st.number_input(
            "Account Balance (€)",
            min_value=0.0,
            max_value=300000.0,
            step=1000.0,
            key="balance"
        )
        estimated_salary = st.number_input(
            "Estimated Salary (€)",
            min_value=0.0,
            max_value=250000.0,
            step=1000.0,
            key="estimated_salary"
        )
        
    with col3:
        st.subheader("Bank Relationship")
        tenure = st.slider(
            "Tenure (Years with Bank)",
            min_value=0,
            max_value=10,
            key="tenure"
        )
        num_of_products = st.selectbox(
            "Number of Products Used",
            options=[1, 2, 3, 4],
            key="num_of_products"
        )
        has_cr_card = st.radio(
            "Has Credit Card?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="has_cr_card"
        )
        is_active_member = st.radio(
            "Is Active Member?",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="is_active_member"
        )
        
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trigger prediction on form submit / click
    st.markdown('<br>', unsafe_allow_html=True)
    
    predict_clicked = st.button(
        "🔮 Analyze Churn Probability",
        type="primary",
        use_container_width=True
    )
    
    # Execution block
    if predict_clicked or 'prediction_ran' in st.session_state:
        st.session_state.prediction_ran = True
        
        # 1. Structure the input values into a DataFrame
        input_dict = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_of_products,
            "HasCrCard": has_cr_card,
            "IsActiveMember": is_active_member,
            "EstimatedSalary": estimated_salary
        }
        
        input_df = pd.DataFrame([input_dict])
        
        # 2. Apply encoders & preprocessors
        try:
            # Map gender using label encoder
            input_df["Gender"] = gender_encoder.transform(input_df["Gender"])
            
            # Map geography using one-hot encoder
            geo_encoded = geo_encoder.transform(input_df[["Geography"]]).toarray()
            geo_encoded_df = pd.DataFrame(
                geo_encoded,
                columns=geo_encoder.get_feature_names_out(["Geography"])
            )
            
            # Concat and select columns in the exact order fitted by the scaler
            preprocessed_df = pd.concat([input_df.drop("Geography", axis=1), geo_encoded_df], axis=1)
            
            expected_cols = [
                "CreditScore", "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography_Germany", "Geography_Spain"
            ]
            preprocessed_df = preprocessed_df[expected_cols]
            
            # Scale features
            scaled_features = scaler.transform(preprocessed_df)
            
            # 3. Model Inference
            prediction_prob = float(model.predict(scaled_features)[0][0])
            churn_risk_percent = prediction_prob * 100
            
            # Display results
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.subheader("📊 Churn Risk Assessment Result")
            
            if prediction_prob > 0.5:
                # High Risk Card
                st.markdown(f"""
                <div class="result-card-high">
                    <div class="metric-label" style="color: #ef4444; font-weight: 700;">🚨 Alert: HIGH CHURN RISK</div>
                    <div class="metric-value" style="color: #ef4444;">{churn_risk_percent:.2f}%</div>
                    <div style="font-size: 16px; font-weight: 500; margin-top: 10px;">
                        The customer exhibits behavioral & financial characteristics that strongly indicate a likelihood to churn.
                    </div>
                    <div class="gauge-container">
                        <div class="gauge-fill gauge-high" style="width: {churn_risk_percent}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Recommendations block
                st.markdown("### 🔑 Retention Strategies & Recommendations")
                recs = []
                if age > 45:
                    recs.append("👵 **Age-related Retention**: Customer is in a mature age segment. Offer personalized wealth management plans, premium retirement consulting services, or custom low-risk investment options.")
                if num_of_products >= 3:
                    recs.append("📦 **Product Saturation Warning**: Customer uses multiple bank products but has high churn risk. Audit account fees and check if they are dissatisfied with the complexity or cost of holding multiple products.")
                if balance < 10000:
                    recs.append("📉 **Low Balance Engagement**: The account balance is low. Propose fee waivers or specialized high-yield savings interest rates to incentivize deposit increases.")
                if is_active_member == 0:
                    recs.append("💤 **Inactivity Re-engagement**: Customer is currently inactive. Launch targeted digital communication campaigns, offer cashback incentives on credit card transactions, or conduct a customer wellness outreach call.")
                if not recs:
                    recs.append("💡 **General Retention**: Schedule an account review with a personal banker to address potential service friction and explore tailored financial solutions.")
                    
                for rec in recs:
                    st.info(rec)
            else:
                # Low Risk Card
                st.markdown(f"""
                <div class="result-card-low">
                    <div class="metric-label" style="color: #10b981; font-weight: 700;">✅ Safe: LOW CHURN RISK</div>
                    <div class="metric-value" style="color: #10b981;">{churn_risk_percent:.2f}%</div>
                    <div style="font-size: 16px; font-weight: 500; margin-top: 10px;">
                        The customer has a stable profile and is likely to remain loyal to the bank.
                    </div>
                    <div class="gauge-container">
                        <div class="gauge-fill gauge-low" style="width: {churn_risk_percent}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🎯 Relationship Nurturing Advice")
                st.success("📈 **Relationship Growth**: The customer is highly loyal. Consider cross-selling high-value financial products (e.g. mortgages, investment portfolios, insurance) to increase average revenue per user.")
                if is_active_member == 1:
                    st.info("⭐ **Loyalty Rewards**: Customer is an active member. Offer loyalty reward tier upgrades or exclusive discounts on affiliate merchant services to sustain high engagement.")
            
        except Exception as e:
            st.error(f"Failed to process input data or calculate prediction: {e}")
            st.warning("Please ensure the inputs match typical ranges.")
            
    # Premium Data Insights section
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander("🔍 Explore Dataset Distribution & Model Performance"):
        st.subheader("Trained Dataset Overview")
        st.write(f"The ANN model was trained on the `Churn_Modelling` dataset which contains `{len(churn_df)}` rows.")
        
        # Display small interactive table of customer demographics
        cols_to_show = ['Surname', 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Exited']
        st.dataframe(churn_df[cols_to_show].head(10), use_container_width=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Show mini statistics
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Total Customer Records", f"{len(churn_df):,}")
        with stats_col2:
            overall_churn = (churn_df['Exited'].mean()) * 100
            st.metric("Overall Churn Rate", f"{overall_churn:.2f}%")
        with stats_col3:
            avg_credit_score = churn_df['CreditScore'].mean()
            st.metric("Average Credit Score", f"{avg_credit_score:.1f}")

else:
    st.warning("System files are missing. Please deploy the files in the directory.")
