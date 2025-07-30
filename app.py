import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb  # Import XGBoost

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Streamlit app title
st.title("Chronic Kidney Disease Prediction with Model Comparison")

# File uploader for dataset upload
data_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if data_file is not None:
    # Read the uploaded dataset
    df = pd.read_csv(data_file)
    st.write("### Uploaded Dataset:")
    st.dataframe(df.head())

    # Preprocessing steps
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)  # Drop ID column if exists

    # Rename columns (ensure consistency)
    df.columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                  'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']

    # Convert columns to numeric where needed
    df['pcv'] = pd.to_numeric(df['pcv'], errors='coerce')
    df['wc'] = pd.to_numeric(df['wc'], errors='coerce')
    df['rc'] = pd.to_numeric(df['rc'], errors='coerce')

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_encoder.fit_transform(df[col])

    # Split dataset
    X = df.drop('classification', axis=1)
    y = df['classification']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X.columns[selector.get_support()]

    # Standardize numerical features
    scaler = StandardScaler()
    X_train_selected = scaler.fit_transform(X_train_selected)
    X_test_selected = scaler.transform(X_test_selected)

    # Train models with fewer trees in Random Forest
    rf_model = RandomForestClassifier(n_estimators=30, random_state=30)  # Reduced to 30 trees
    rf_model.fit(X_train_selected, y_train)
    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_selected, y_train)

    # Predictions
    rf_predictions = rf_model.predict(X_test_selected)
    xgb_predictions = xgb_model.predict(X_test_selected)

    # Model evaluation
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)

    # Display accuracies
    st.write("### Model Accuracies")
    st.write(f"**Random Forest Accuracy:** {rf_accuracy:.2f}")
    st.write(f"**XGBoost Accuracy:** {xgb_accuracy:.2f}")

    # Confusion Matrix Visualization
    st.write("### Confusion Matrix")

    # Random Forest Confusion Matrix
    st.write("#### Random Forest Confusion Matrix")
    rf_cm = confusion_matrix(y_test, rf_predictions)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Random Forest Confusion Matrix')
    st.pyplot(fig)

    # XGBoost Confusion Matrix
    st.write("#### XGBoost Confusion Matrix")
    xgb_cm = confusion_matrix(y_test, xgb_predictions)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Reds', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('XGBoost Confusion Matrix')
    st.pyplot(fig)

    # Add a section to display the chart counts from the images
    st.write("### Chart Counts from Images")

    # Updated counts for Random Forest and XGBoost based on the image
    st.write("**Random Forest Counts:**")
    rf_counts = [69, 2, 0, 7, 42]  # Updated counts for Random Forest
    st.write(rf_counts)

    st.write("**XGBoost Counts:**")
    xgb_counts = [70, 60, 50, 40, 30, 20, 10, 0]  # Updated counts for XGBoost
    st.write(xgb_counts)

    # Visualize the updated counts using a bar plot
    st.write("### Visualization of Counts")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Random Forest counts
    ax.bar(np.arange(len(rf_counts)) - 0.2, rf_counts, width=0.4, color='blue', label='Random Forest')

    # Plot XGBoost counts
    ax.bar(np.arange(len(xgb_counts)) + 0.2, xgb_counts, width=0.4, color='red', label='XGBoost', alpha=0.7)

    # Customize the plot
    ax.set_xticks(np.arange(max(len(rf_counts), len(xgb_counts))))
    ax.set_xlabel("Index")
    ax.set_ylabel("Counts")
    ax.set_title("Random Forest vs XGBoost Counts")
    ax.legend()
    st.pyplot(fig)

    # Model selection for prediction
    model_choice = st.selectbox("Select Model for Prediction", ["Random Forest", "XGBoost"])

    # Individual Record Entry
    st.write("### Predict CKD for an Individual Record")
    individual_data = {}
    for col in X.columns:
        if col in df.select_dtypes(include=['object']).columns:
            unique_values = df[col].unique()
            individual_data[col] = st.selectbox(f"{col}", options=unique_values)
        else:
            individual_data[col] = st.number_input(f"{col}", value=float(df[col].median()))

    if st.button("Predict CKD"):
        individual_df = pd.DataFrame([individual_data])
        for col in individual_df.select_dtypes(include='object').columns:
            individual_df[col] = label_encoder.transform(individual_df[col])
        individual_df = individual_df[selected_features]
        individual_df = scaler.transform(individual_df)
        prediction = rf_model.predict(individual_df)[0] if model_choice == "Random Forest" else xgb_model.predict(individual_df)[0]
        
        # Display prediction result in a pop-up dark
        with st.popover("Prediction Result"):
            st.write("### Individual Prediction Result")
            st.write(f"**Prediction:** {'CKD' if prediction == 1 else 'No CKD'}")
            st.write("**Model Used:**", model_choice)

    # Predict CKD for all records
    st.write("### Predict CKD for All Records")
    if st.button("Predict for All Records"):
        all_predictions = rf_model.predict(X_test_selected) if model_choice == "Random Forest" else xgb_model.predict(X_test_selected)
        X_test_with_predictions = X_test.copy()
        X_test_with_predictions['Prediction'] = ["CKD" if pred == 1 else "No CKD" for pred in all_predictions]
        st.write(X_test_with_predictions)
else:
    st.write("Please upload a dataset to proceed.")