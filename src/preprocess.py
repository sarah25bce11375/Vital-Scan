import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Load the heart disease dataset."""
    df = pd.read_csv(filepath)
    return df

def check_data(df):
    """Print basic info about the dataset."""
    print("Shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nData types:\n", df.dtypes)
    print("\nTarget distribution:\n", df['target'].value_counts())

def preprocess(df):
    """
    Clean and preprocess the dataset.
    Returns X (features), y (labels), scaler.
    """
    df = df.dropna()

    # Rename target column if needed (some versions use 'condition')
    if 'condition' in df.columns:
        df = df.rename(columns={'condition': 'target'})

    X = df.drop('target', axis=1)
    y = df['target']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler

def get_feature_names():
    """Return the expected feature names for the UCI Heart Disease dataset."""
    return [
        'age', 'sex', 'cp', 'trestbps', 'chol',
        'fbs', 'restecg', 'thalach', 'exang',
        'oldpeak', 'slope', 'ca', 'thal'
    ]  # ← closing bracket and parenthesis were missing