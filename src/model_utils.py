import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return metrics."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred,average='macro'),
        'recall': recall_score(y_test, y_pred,average='macro'),
        'f1_score': f1_score(y_test, y_pred,average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    return metrics, y_pred

def save_model(model, scaler, output_dir, model_filename='model.joblib', scaler_filename='scaler.joblib'):
    """Save model and scaler to files."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and scaler
    model_path = os.path.join(output_dir, model_filename)
    scaler_path = os.path.join(output_dir, scaler_filename)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model_path, scaler_path

def load_model(model_path, scaler_path=None):
    """Load model and optionally scaler from files."""
    model = joblib.load(model_path)
    
    scaler = None
    if scaler_path:
        scaler = joblib.load(scaler_path)
    
    return model, scaler
