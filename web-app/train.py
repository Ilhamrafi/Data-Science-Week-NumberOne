from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, roc_curve, auc

path = 'Telco_customer_churn_adapted_v2.xlsx'

def train():
    data = pd.read_excel(path, index_col='Customer ID')
    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])

    # Splitting the data
    X = data.drop('Churn Label', axis=1)
    y = data['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Models
    models = {
        "Support Vector Machine": SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Training and Evaluating Models
    results = {}

    for name, model in models.items():
        # Train the model
        model.fit(X_train_scaled, y_train)
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        # Store the results
        results[name]= {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC AUC': roc_auc}

    results_df = pd.DataFrame(results).T
    
    feature_importance = feature_importances(models, X)
    return results_df, y_train, feature_importance

def train_smote():
    data = pd.read_excel(path, index_col='Customer ID')

    le = LabelEncoder()
    for column in data.columns:
        if data[column].dtype == 'object':
            data[column] = le.fit_transform(data[column])

    # Splitting the data
    X = data.drop('Churn Label', axis=1)
    y = data['Churn Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # Models
    models = {
        "Support Vector Machine": SVC(kernel='linear'),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    # Training and Evaluating Models
    results = {}

    for name, model in models.items():
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        # Store the results
        results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC AUC': roc_auc}

    results_df = pd.DataFrame(results).T

    feature_importance = feature_importances(models, X)

    return results_df, y_train_resampled, feature_importance

def visualisasi_label(y_train, y_train_resampled):
    # Data
    labels = ['Class 0', 'Class 1']  # Ganti dengan label kelas yang sesuai dengan dataset Anda

    # Hitung frekuensi kelas untuk y_train
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    freq_train = dict(zip(unique_train, counts_train))

    # Hitung frekuensi kelas untuk y_train_smote
    unique_smote, counts_smote = np.unique(y_train_resampled, return_counts=True)
    freq_smote = dict(zip(unique_smote, counts_smote))

    # Plot histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Plot untuk y_train
    colors = sns.color_palette("coolwarm", len(labels))
    ax1.bar(labels, [freq_train.get(0, 0), freq_train.get(1, 0)], color=colors)
    ax1.set_title('y_train (Original)')
    ax1.set_ylabel('Frequency')

    # Plot untuk y_train_smote
    ax2.bar(labels, [freq_smote.get(0, 0), freq_smote.get(1, 0)], color=colors)
    ax2.set_title('y_train_smote (After SMOTE)')

    return st.pyplot(fig)

def visualisasi_perbadingan(results_df):
    sns.set_palette("coolwarm")

    # Creating a more concise visualization
    fig = plt.figure(figsize=(15, 8))

    # Melting the dataframe to a long format for easier plotting
    long_df = results_df.reset_index().melt(id_vars="index")
    long_df.rename(columns={'index': 'Model', 'variable': 'Metric', 'value': 'Score'}, inplace=True)

    # Plotting
    sns.barplot(x='Metric', y='Score', hue='Model', data=long_df)
    plt.title('Model Performance Comparison')
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Score')
    plt.legend(title='Model')

    return st.pyplot(fig)

def feature_importances(models, X):
    # Inisialisasi DataFrame kosong
    feature_importance_df = pd.DataFrame()

    # Loop through models and extract feature importance
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = model.coef_[0]
        else:
            feature_importance = None

        # Menambahkan kolom dengan nama model dan nilai feature importance
        feature_importance_df[name] = feature_importance

    # Menambahkan kolom 'Feature' dengan nama fitur-fitur dari dataset
    feature_importance_df['Feature'] = X.columns

    # Melting the DataFrame for easier plotting
    feature_importance_melted = feature_importance_df.melt(id_vars='Feature', var_name='Model', value_name='Importance')
    
    return feature_importance_melted


def plot_feature_importance(feature_importance_melted):
    fig = plt.figure(figsize=(10, 5))
    sns.barplot(x='Importance', y='Feature', hue='Model', data=feature_importance_melted)
    plt.title('Feature Importance Comparison')
    plt.xlabel('Importance')
    plt.ylabel('Feature')

    return st.pyplot(fig)
