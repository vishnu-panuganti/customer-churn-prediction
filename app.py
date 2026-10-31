import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)
# --------------------------------
# CUSTOM DASHBOARD STYLE
# --------------------------------
# --------------------------------
# DARK DASHBOARD STYLE
# --------------------------------

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #0e1117;
        color: #f1f5f9;
    }

    /* Main content */
    .main {
        background-color: #0e1117;
    }

    /* Headings */
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-weight: 700;
    }

    /* Normal text */
    p, label, span {
        color: #cbd5e1;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    /* Metric labels */
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    /* Metric values */
    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #11161d;
        border-right: 1px solid #30363d;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    /* Select boxes */
    div[data-baseweb="select"] > div {
        background-color: #161b22;
        border-color: #30363d;
        color: #f8fafc;
    }

    /* Number input */
    div[data-testid="stNumberInput"] input {
        background-color: #161b22;
        color: #f8fafc;
        border-color: #30363d;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        height: 45px;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Info boxes */
    div[data-testid="stAlert"] {
        background-color: #161b22;
        border-color: #30363d;
    }

    /* Dataframes / tables */
    div[data-testid="stDataFrame"] {
        background-color: #161b22;
    }

    /* Horizontal lines */
    hr {
        border-color: #30363d;
    }

</style>
""", unsafe_allow_html=True)

# --------------------------------
# LOAD DATA
# --------------------------------

df = pd.read_csv("data/tele.csv")

# Keep original churn for actual result
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Load trained model
model = joblib.load("model/churn_model.pkl")


# --------------------------------
# SIDEBAR
# --------------------------------

st.sidebar.title("📊 Churn Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🔮 Predict Churn",
        "🤖 Model Performance"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Customer Churn Prediction")
st.sidebar.write("AI Internship Project")


# ==============================================
# OVERVIEW PAGE
# ==============================================

if page == "📊 Overview":

    st.title("📊 Customer Churn Analytics")

    st.write(
        "Analyze customer churn patterns and identify customers "
        "who are at risk of leaving."
    )

    total_customers = len(df)
    churned_customers = df["Churn"].sum()

    churn_rate = (
        churned_customers / total_customers
    ) * 100

    best_model = "Logistic Regression"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Churned Customers",
            f"{churned_customers:,}"
        )

    with col3:
        st.metric(
            "Churn Rate",
            f"{churn_rate:.1f}%"
        )

    with col4:
        st.metric(
            "Best Model",
            best_model
        )

    st.markdown("---")

    st.subheader("Customer Churn Distribution")

    churn_chart = df["Churn"].value_counts()

    churn_chart.index = [
        "Stayed",
        "Churned"
    ]

    st.bar_chart(churn_chart)

    st.subheader("Churn by Contract Type")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"]
    )

    contract_churn.columns = [
        "Stayed",
        "Churned"
    ]

    st.bar_chart(contract_churn)

    st.subheader("Churn by Internet Service")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"]
    )

    internet_churn.columns = [
        "Stayed",
        "Churned"
    ]

    st.bar_chart(internet_churn)


# ==============================================
# PREDICTION PAGE
# ==============================================

elif page == "🔮 Predict Churn":

    st.title("🔮 Customer Churn Prediction")

    st.write(
        "Predict churn for a new customer or test the model "
        "using an existing customer from the dataset."
    )

    # ==========================================
    # REAL CUSTOMER TEST
    # ==========================================

    st.subheader("🔍 Test an Existing Customer")

    st.write(
        "Select a real customer from the dataset and compare "
        "the model prediction with the actual result."
    )

    customer_id = st.selectbox(
        "Select Customer ID",
        df["customerID"].tolist()
    )

    selected_customer = df[
        df["customerID"] == customer_id
    ].iloc[0]

    st.markdown("### Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Tenure",
            f"{selected_customer['tenure']} months"
        )

    with col2:
        st.metric(
            "Monthly Charges",
            f"${selected_customer['MonthlyCharges']:.2f}"
        )

    with col3:
        st.metric(
            "Contract",
            selected_customer["Contract"]
        )

    # --------------------------------
    # PREPARE CUSTOMER FOR MODEL
    # --------------------------------

    customer_input = selected_customer.drop(
        "Churn"
    ).to_frame().T

    customer_input = customer_input.drop(
        "customerID",
        axis=1
    )

    # Convert TotalCharges to numeric
    customer_input["TotalCharges"] = pd.to_numeric(
        customer_input["TotalCharges"],
        errors="coerce"
    )

    # --------------------------------
    # PREDICTION
    # --------------------------------

    if st.button(
        "🔍 Test This Customer",
        use_container_width=True
    ):

        prediction = model.predict(
            customer_input
        )[0]

        probability = model.predict_proba(
            customer_input
        )[0][1]

        probability_percent = probability * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Churn Probability",
                f"{probability_percent:.1f}%"
            )

        with col2:

            if probability >= 0.70:

                st.error(
                    "🔴 HIGH RISK"
                )

            elif probability >= 0.40:

                st.warning(
                    "🟡 MEDIUM RISK"
                )

            else:

                st.success(
                    "🟢 LOW RISK"
                )

        # --------------------------------
        # ACTUAL RESULT
        # --------------------------------

        st.subheader("Actual Result")

        actual = selected_customer["Churn"]

        if actual == 1:

            st.error(
                "🔴 Actual Result: Customer CHURNED"
            )

        else:

            st.success(
                "🟢 Actual Result: Customer STAYED"
            )

        # --------------------------------
        # CHECK WHETHER CORRECT
        # --------------------------------

        if prediction == actual:

            st.success(
                "✅ Model prediction is correct!"
            )

        else:

            st.warning(
                "⚠️ Model prediction does not match "
                "the actual result."
            )

    st.markdown("---")

    # ==========================================
    # MANUAL PREDICTION
    # ==========================================

    st.subheader("🧑‍💻 Predict a New Customer")

    st.write(
        "Enter customer information manually."
    )

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            key="gender"
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            key="senior"
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"],
            key="partner"
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"],
            key="dependents"
        )

        tenure = st.number_input(
            "Tenure (months)",
            min_value=0,
            max_value=100,
            value=12,
            key="tenure"
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"],
            key="phone"
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
            key="lines"
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            key="internet"
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
            key="security"
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
            key="backup"
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
            key="device"
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
            key="support"
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
            key="tv"
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
            key="movies"
        )

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ],
            key="contract"
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            key="paperless"
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            key="payment"
        )

    st.subheader("Billing Information")

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            key="monthly"
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=float(monthly_charges * tenure),
            key="total"
        )

    if st.button(
        "🔮 Predict Churn",
        use_container_width=True
    ):

        customer = pd.DataFrame({

            "gender": [gender],
            "SeniorCitizen": [senior_citizen],
            "Partner": [partner],
            "Dependents": [dependents],
            "tenure": [tenure],
            "PhoneService": [phone_service],
            "MultipleLines": [multiple_lines],
            "InternetService": [internet_service],
            "OnlineSecurity": [online_security],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_protection],
            "TechSupport": [tech_support],
            "StreamingTV": [streaming_tv],
            "StreamingMovies": [streaming_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless_billing],
            "PaymentMethod": [payment_method],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges]
        })

        prediction = model.predict(customer)[0]

        probability = model.predict_proba(
            customer
        )[0][1]

        probability_percent = probability * 100

        st.markdown("---")

        st.subheader("Prediction Result")

        st.metric(
            "Churn Probability",
            f"{probability_percent:.1f}%"
        )

        if probability >= 0.70:

            st.error(
                "🔴 HIGH RISK — Customer likely to churn"
            )

        elif probability >= 0.40:

            st.warning(
                "🟡 MEDIUM RISK — Customer may churn"
            )

        else:

            st.success(
                "🟢 LOW RISK — Customer likely to stay"
            )


# ==============================================
# MODEL PERFORMANCE
# ==============================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Comparison of machine learning models "
        "evaluated for customer churn prediction."
    )

    st.markdown("---")

    # --------------------------------
    # MODEL RESULTS
    # --------------------------------

    results = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest"
        ],

        "Accuracy": [
            0.7257,
            0.7335,
            0.7676
        ],

        "Precision": [
            0.4901,
            0.4992,
            0.5535
        ],

        "Recall": [
            0.7968,
            0.7995,
            0.6497
        ],

        "F1 Score": [
            0.6069,
            0.6146,
            0.5978
        ],

        "ROC-AUC": [
            0.8351,
            0.8182,
            0.8146
        ]
    })

    # --------------------------------
    # DISPLAY TABLE
    # --------------------------------

    st.subheader("Model Comparison")

    display_results = results.copy()

    for column in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------
    # BEST MODEL
    # --------------------------------

    st.markdown("---")

    st.subheader("🏆 Selected Model")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Selected Model",
            "Logistic Regression"
        )

    with col2:

        st.metric(
            "ROC-AUC",
            "83.51%"
        )

    st.info(
        "Logistic Regression was selected because it achieved "
        "the highest ROC-AUC (83.51%) while maintaining a high "
        "recall of 79.68%, which is important for identifying "
        "customers who may churn."
    )

    # --------------------------------
    # MODEL INSIGHTS
    # --------------------------------

    st.markdown("---")

    st.subheader(" Model Insights")

    st.write(
        """
        **Logistic Regression**
        
        Best ROC-AUC and strong recall. It provides a good
        balance for identifying customers who are likely to churn.

        **Decision Tree**
        
        Achieved the highest recall and F1 score, meaning it
        identified a large proportion of actual churners.

        **Random Forest**
        
        Achieved the highest accuracy and precision, but its
        recall was lower than Logistic Regression and Decision Tree.
        """
    )
