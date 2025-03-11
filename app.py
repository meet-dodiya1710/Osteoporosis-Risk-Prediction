import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import time

# Streamlit App Title with Enhanced Theme
st.set_page_config(page_title="Osteoporosis Risk Prediction", layout="wide")

st.markdown("<h1 class='main-title'>🦴 Osteoporosis Risk Prediction</h1>", unsafe_allow_html=True)

# Sidebar for Model Selection
st.sidebar.header("🔧 Model Settings")
model_choice = st.sidebar.selectbox("🧠 Choose Model", ["Logistic Regression", "Random Forest", "Decision Tree", "SVM"])

# Upload CSV file
data_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
if data_file is not None:
    data = pd.read_csv(data_file)
    
    # Dataset Overview
    st.markdown("<h2 class='sub-title'>📊 Dataset Overview</h2>", unsafe_allow_html=True)
    st.write(f"🔹 *Shape:* {data.shape[0]} rows × {data.shape[1]} columns")
    st.dataframe(data.head())
    
    # Missing Values Info
    st.markdown("<h2 class='sub-title'>🚨 Missing Values Information</h2>", unsafe_allow_html=True)
    missing_values = data.isnull().sum()
    st.dataframe(missing_values[missing_values > 0])
    if missing_values.sum() > 0:
        st.warning("⚠ Missing values detected. Filling with mode.")
        for col in missing_values.index:
            data[col].fillna(data[col].mode()[0], inplace=True)
        st.success("✅ Missing values filled.")
    else:
        st.success("✅ No missing values found.")
    
    # Categorical Column Statistics
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    st.markdown("<h2 class='sub-title'>📊 Categorical Column Summary</h2>", unsafe_allow_html=True)
    categorical_summary = data[categorical_cols].describe().transpose()
    st.dataframe(categorical_summary)
    
    # Feature Engineering
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    selected_features = [col for col in data.columns if col not in ['Id', 'Osteoporosis']]
    X = data[selected_features]
    y = data["Osteoporosis"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    if "model" not in st.session_state:
        st.session_state.model = None
    
    # Model Training
    st.sidebar.subheader("🚀 Train the Model")
    if st.sidebar.button("Start Training"):
        # Placeholder for loading animation in the middle of the screen
        loading_placeholder = st.empty()

        # Display loading animation in the center
        with loading_placeholder.container():
            st.markdown("<h2 style='text-align: center;'>⏳ Training in progress... Please wait.</h2>", unsafe_allow_html=True)
            st.markdown("<div style='text-align: center; font-size: 50px;'>🔄</div>", unsafe_allow_html=True)
            time.sleep(2)  # Simulate training delay

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(probability=True)
        }

        model = models[model_choice]
        model.fit(X_train, y_train)
        st.session_state.model = model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Remove loading animation and show results
        loading_placeholder.empty()
        st.success("✅ Model training completed!")

        st.markdown(f"<h2 class='sub-title'>🏆 {model_choice} Results</h2>", unsafe_allow_html=True)
        st.write(f"🔹 Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df.style.format(precision=2))
        
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
        
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend()
            st.pyplot(fig)
    
    # User Input for Prediction
    st.markdown("<h2 class='sub-title'>🧑‍⚕ Predict Osteoporosis Risk for a New Patient</h2>", unsafe_allow_html=True)
    input_data = {}
    
    for feature in selected_features:
        if feature in categorical_cols:
            input_data[feature] = st.selectbox(f"Select {feature}", label_encoders[feature].classes_)
        elif feature.lower() == "age":  # Ensuring age is taken as an integer
            input_data[feature] = st.number_input(f"Enter {feature}", min_value=int(data[feature].min()), 
                                                  max_value=int(data[feature].max()), 
                                                  value=int(data[feature].mean()), step=1)
        else:
            input_data[feature] = st.number_input(f"Enter {feature}", min_value=float(data[feature].min()), 
                                                  max_value=float(data[feature].max()), 
                                                  value=float(data[feature].mean()))
    
    if st.button("🔍 Predict"):
        if st.session_state.model is None:
            st.error("🚨 No trained model found! Please train a model first.")
        else:
            for col in categorical_cols:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            user_input_df = pd.DataFrame([input_data])
            user_input_scaled = scaler.transform(user_input_df)
            prediction = st.session_state.model.predict(user_input_scaled)[0]
            result = "🟢 Low Risk" if prediction == 0 else "🔴 High Risk"
            st.success(f"### Prediction: {result}")
