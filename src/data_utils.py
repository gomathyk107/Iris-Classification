"""
Data utility functions for preparing and processing diabetic dataset.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(data_path):
    """Load data from CSV and perform basic cleaning."""
    # Load data
    df = pd.read_csv(data_path)
    
    # Basic cleaning
    # Replace missing values with median for numeric columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    return df

def prepare_features_and_target(df, target_column='Outcome'):
    """Split dataframe into features and target."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Split data into train/test sets and scale features."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
