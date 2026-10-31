# 📊 Customer Churn Prediction

> An end-to-end machine learning project that predicts customer churn probability and identifies customers who may be at risk of leaving.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)](https://pandas.pydata.org/)

---

## 📌 Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help companies take preventive retention actions.

This project uses machine learning to analyze customer information and predict whether a customer is likely to churn.

The project includes:

- Data cleaning and preprocessing
- Exploratory churn analysis
- Multiple machine learning models
- Model performance comparison
- Churn probability prediction
- Customer risk classification
- Existing-customer prediction testing
- Interactive Streamlit dashboard

---

## 🎯 Objectives

The main objectives of this project are to:

- Analyze customer churn patterns
- Identify important customer characteristics associated with churn
- Build classification models for churn prediction
- Compare multiple machine learning algorithms
- Select a suitable final model
- Provide an easy-to-use prediction dashboard

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and preprocessing |
| Joblib | Model serialization |
| Streamlit | Interactive dashboard |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Git & GitHub | Version control |

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   └── tele.csv
│
├── model/
│   └── churn_model.pkl
│
├── notebooks/
│
├── app.py
├── train_model.py
├── test.py
├── .gitignore
└── README.md
```

---

## 📊 Dataset

The project uses the **Telco Customer Churn dataset**, containing information about telecom customers and whether they eventually churned.

### Dataset Information

- Original records: **7,043**
- Records after cleaning: **7,032**
- Features used for prediction: **19**
- Target variable: **Churn**

### Major Features

The dataset contains information related to:

- Customer demographics
- Partner and dependent status
- Customer tenure
- Phone services
- Internet services
- Online security
- Online backup
- Device protection
- Technical support
- Streaming services
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges

---

## 🔄 Machine Learning Workflow

```text
                 ┌─────────────────┐
                 │   Telco Dataset │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │  Data Cleaning  │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Feature         │
                 │ Preprocessing   │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Train / Test    │
                 │ Split           │
                 └────────┬────────┘
                          ↓
            ┌─────────────┼─────────────┐
            ↓             ↓             ↓
       Logistic       Decision       Random
      Regression        Tree         Forest
            └─────────────┼─────────────┘
                          ↓
                 ┌─────────────────┐
                 │ Model Evaluation│
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Final Model     │
                 └────────┬────────┘
                          ↓
                 ┌─────────────────┐
                 │ Streamlit       │
                 │ Dashboard       │
                 └─────────────────┘
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Converted `TotalCharges` into numeric format.
3. Handled missing values.
4. Removed the `customerID` column from model features.
5. Converted the target variable:
   - `No → 0`
   - `Yes → 1`
6. Split the dataset into training and testing sets.
7. Applied `StandardScaler` to numerical features.
8. Applied `OneHotEncoder` to categorical features.
9. Combined preprocessing and model training using a Scikit-learn Pipeline.

### Train/Test Split

```text
Training Data : 80%
Testing Data  : 20%
Random State  : 42
Stratification: Yes
```

---

## 🤖 Models Evaluated

Three classification algorithms were trained and evaluated:

### 1. Logistic Regression

Used as a baseline classification model and provided strong churn detection performance.

### 2. Decision Tree

Used to capture non-linear relationships between customer characteristics and churn.

### 3. Random Forest

An ensemble learning algorithm used to improve predictive performance by combining multiple decision trees.

---

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 72.57% | 49.01% | 79.68% | 60.69% | **83.51%** |
| Decision Tree | 73.35% | 49.92% | **79.95%** | **61.46%** | 81.82% |
| Random Forest | **76.76%** | **55.35%** | 64.97% | 59.78% | 81.46% |

### 🏆 Selected Model

**Logistic Regression** was selected as the final model.

The selection was based primarily on its:

- **83.51% ROC-AUC**
- **79.68% recall**
- Strong ability to distinguish between customers who churn and customers who stay

Recall is particularly relevant in this project because failing to identify an actual churner can result in a missed customer-retention opportunity.

The trained model is stored at:

```text
model/churn_model.pkl
```

---

# 🖥️ Interactive Dashboard

The project includes a dark-themed Streamlit dashboard designed to provide an easy way to explore customer churn and generate predictions.

## 📊 1. Overview

The Overview section provides a high-level view of the dataset.

It displays:

- Total customers
- Churned customers
- Overall churn rate
- Selected machine learning model
- Customer churn distribution
- Churn by contract type
- Churn by internet service

---

## 🔮 2. Predict Churn

The prediction section supports two approaches.

### Existing Customer

A user can select a real customer from the dataset and view:

- Customer tenure
- Monthly charges
- Contract type
- Predicted churn probability
- Risk level
- Actual churn result
- Whether the prediction matches the actual result

### New Customer

Users can enter customer details manually and receive:

- Churn probability
- Predicted churn status
- Risk classification

### Risk Levels

```text
🟢 LOW RISK      < 40%

🟡 MEDIUM RISK   40% – 69.99%

🔴 HIGH RISK     ≥ 70%
```

These thresholds are application-level risk categories used to make the model output easier to interpret.

---

## 🤖 3. Model Performance

The dashboard also provides a comparison of:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

This allows users to understand why the final model was selected.

---

## ▶️ Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/vishnu-panuganti/customer-churn-prediction.git
```

### 2. Enter the project directory

```bash
cd customer-churn-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install pandas numpy scikit-learn joblib streamlit matplotlib seaborn
```

### 6. Run the application

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

---

## 💡 Key Insights

The analysis highlights several useful observations:

- Contract type is an important factor associated with customer churn.
- Customer tenure is an important variable in churn prediction.
- Monthly and total charges contribute to the prediction.
- Service-related features can provide useful information for identifying potential churners.
- Machine learning can help businesses identify customers who may benefit from retention strategies.

---

## 🚀 Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Cross-validation
- Feature importance visualization
- SHAP-based explainability
- Automated retention recommendations
- Model monitoring
- Cloud deployment
- Integration with a live customer database
- Real-time churn prediction API
- Automated customer segmentation

---

## 📸 Dashboard Preview

> Add screenshots of the Streamlit dashboard here after capturing them.

Example:

```text
docs/
├── overview.png
├── prediction.png
└── model-performance.png
```

Then they can be displayed in this section using:

```markdown
### Overview

![Dashboard Overview](docs/overview.png)

### Churn Prediction

![Churn Prediction](docs/prediction.png)

### Model Performance

![Model Performance](docs/model-performance.png)
```

---

## 🎓 Project Purpose

This project was developed as a practical machine learning project to demonstrate skills in:

- Data preprocessing
- Classification
- Model evaluation
- Machine learning pipelines
- Python development
- Data visualization
- Streamlit application development
- Git and GitHub

---

## 👨‍💻 Author

### Vishnu Panuganti

Computer Science Engineering Student

Interested in:

**Machine Learning • AI • Cloud Computing • Cybersecurity • Software Development**

---

## ⭐ If you found this project useful

Feel free to explore the repository, experiment with the model, and improve the application.

---

## 📄 License

This project is intended for educational and portfolio purposes.
