#Import required libraries 
import os 
import sys 
import numpy as np 
import pandas as pd 
import joblib 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow 
import mlflow.sklearn

#Add the src directory to the path for importing our utility modules
#This assumes the script is run from a directory where '../' correctly points to the project root.

# sys.path.append('../')

# Get the directory of the current script
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'src' is a direct subdirectory of the project root, and this script is in the project root.
_project_root_dir = os.path.abspath(_script_dir)
# Add the 'src' directory to the beginning of the Python path
sys.path.insert(0, os.path.join(_project_root_dir, 'src'))

# Now you can import from src
from src.data_utils import load_and_clean_data, prepare_features_and_target, split_and_scale_data

#If you have specific functions in model_utils that are critical and not replaced by MLflow, uncomment:

from src.model_utils import evaluate_model, save_model, load_model

def run_training_pipeline(processed_data_dir, model_dir): 
  """ Encapsulates the entire model training, hyperparameter tuning, and best model registration pipeline using MLflow.
  Args:
    processed_data_dir (str): Path to the directory containing processed data (NPZ files, scaler, feature names).
    model_dir (str): Path to the directory where MLflow runs and models will be stored.

  Returns:
      str: MLflow URI of the best registered model (e.g., "models:/IrisBestModel/latest"),
          or None if training fails.
  """
  # Ensure model directory exists
  os.makedirs(model_dir, exist_ok=True)

  # Load the prepared data
  train_data_path = os.path.join(processed_data_dir, 'train_data.npz')
  test_data_path = os.path.join(processed_data_dir, 'test_data.npz')

  try:
      train_data = np.load(train_data_path)
      test_data = np.load(test_data_path)
  except FileNotFoundError as e:
      print(f"Error loading processed data: {e}. Ensure '{processed_data_dir}' contains 'train_data.npz' and 'test_data.npz'.")
      return None

  X_train = train_data['X']
  y_train = train_data['y']
  X_test = test_data['X']
  y_test = test_data['y']

  # Load feature names for reference
  feature_names_path = os.path.join(processed_data_dir, 'feature_names.joblib')
  try:
      feature_names = joblib.load(feature_names_path)
  except FileNotFoundError as e:
      print(f"Warning: Feature names file not found at {feature_names_path}. Proceeding without it. {e}")
      feature_names = [] # Default to empty list if not found

  # Load the scaler (optional, if needed for inference or further processing, but not directly used in training)
  scaler_path = os.path.join(processed_data_dir, 'scaler.joblib')
  try:
      scaler = joblib.load(scaler_path)
  except FileNotFoundError as e:
      print(f"Warning: Scaler file not found at {scaler_path}. Proceeding without it. {e}")
      scaler = None # Default to None if not found

  print(f"Loaded training data: {X_train.shape[0]} samples with {X_train.shape[1]} features")
  print(f"Loaded testing data: {X_test.shape[0]} samples with {X_test.shape[1]} features")
  if feature_names:
      print(f"Feature names: {feature_names}")

  # Set MLflow tracking URI and experiment
  mlflow_tracking_uri = f"file:{model_dir}/mlruns"
  mlflow.set_tracking_uri(mlflow_tracking_uri)
  print(f"MLflow tracking URI set to: {mlflow_tracking_uri}")

  best_f1 = -1.0  # Initialize with a value lower than any possible F1 score
  best_model = None
  best_run_id = None
  best_model_name = ""

  # --- Train Logistic Regression (Base) ---
  mlflow.set_experiment("Iris_Model_Tracking")
  with mlflow.start_run(run_name="LogisticRegression_Base") as run:
      lr = LogisticRegression(max_iter=200, random_state=45)
      lr.fit(X_train, y_train)
      y_pred = lr.predict(X_test)

      acc = accuracy_score(y_test, y_pred)
      precision = precision_score(y_test, y_pred, average="macro")
      recall = recall_score(y_test, y_pred, average="macro")
      f1 = f1_score(y_test, y_pred, average="macro")

      mlflow.log_param("max_iter", 200)
      mlflow.log_metric("accuracy", acc)
      mlflow.log_metric("Precision", precision)
      mlflow.log_metric("Recall", recall)
      mlflow.log_metric("f1_score", f1)

      # Log model artifact, but won't register as the final best model yet
      mlflow.sklearn.log_model(lr, "model_lr_base_artifact")

      print(f"Logistic Regression (Base) - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

      if f1 > best_f1:
          best_f1 = f1
          best_model = lr
          best_run_id = run.info.run_id
          best_model_name = "LogisticRegression_Base"

  # --- Logistic Regression Hyperparameter Tuning ---
  mlflow.set_experiment("LogReg_Hyperparam_Tuning")
  param_grid_lr = {
      'C': [0.01, 0.1, 1, 10],
      'penalty': ['l1', 'l2'],
      'solver': ['liblinear']
  }

  with mlflow.start_run(run_name="LogReg_GridSearch_Metrics") as run:
      grid_lr = GridSearchCV(
          LogisticRegression(max_iter=500, random_state=45),
          param_grid_lr,
          cv=5,
          scoring='accuracy',
          n_jobs=-1
      )
      grid_lr.fit(X_train, y_train)

      best_model_lr_tuned = grid_lr.best_estimator_
      y_pred_lr_tuned = best_model_lr_tuned.predict(X_test)

      acc_lr_tuned = accuracy_score(y_test, y_pred_lr_tuned)
      prec_lr_tuned = precision_score(y_test, y_pred_lr_tuned, average='macro')
      rec_lr_tuned = recall_score(y_test, y_pred_lr_tuned, average='macro')
      f1_lr_tuned = f1_score(y_test, y_pred_lr_tuned, average='macro')

      mlflow.log_params(grid_lr.best_params_)
      mlflow.log_metric("accuracy", acc_lr_tuned)
      mlflow.log_metric("precision", prec_lr_tuned)
      mlflow.log_metric("recall", rec_lr_tuned)
      mlflow.log_metric("f1_score", f1_lr_tuned)

      mlflow.sklearn.log_model(best_model_lr_tuned, "best_model_lr_tuned_artifact")

      print("Logistic Regression (Tuned) - Best Params:", grid_lr.best_params_)
      print(f"Logistic Regression (Tuned) - Accuracy: {acc_lr_tuned:.4f} | F1: {f1_lr_tuned:.4f}")

      if f1_lr_tuned > best_f1:
          best_f1 = f1_lr_tuned
          best_model = best_model_lr_tuned
          best_run_id = run.info.run_id
          best_model_name = "LogisticRegression_Tuned"

  # --- Train Random Forest (Base) ---
  mlflow.set_experiment("Iris_Model_Tracking")
  with mlflow.start_run(run_name="RandomForest_Base") as run:
      rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
      rf.fit(X_train, y_train)
      y_pred = rf.predict(X_test)

      acc = accuracy_score(y_test, y_pred)
      f1 = f1_score(y_test, y_pred, average="macro")

      mlflow.log_param("n_estimators", 100)
      mlflow.log_param("max_depth", 4)
      mlflow.log_metric("accuracy", acc)
      mlflow.log_metric("f1_score", f1)

      mlflow.sklearn.log_model(rf, "model_rf_base_artifact")

      print(f"Random Forest (Base) - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

      if f1 > best_f1:
          best_f1 = f1
          best_model = rf
          best_run_id = run.info.run_id
          best_model_name = "RandomForest_Base"

  # --- Random Forest Hyperparameter Tuning ---
  mlflow.set_experiment("RandomForest_Hyperparam_Tuning")
  param_grid_rf = {
      'n_estimators': [50, 100, 150],
      'max_depth': [3, 5, 10],
      'min_samples_split': [2, 4]
  }

  with mlflow.start_run(run_name="RandomForest_GridSearch_Metrics") as run:
      grid_rf = GridSearchCV(
          RandomForestClassifier(random_state=42),
          param_grid_rf,
          cv=5,
          scoring='accuracy',
          n_jobs=-1
      )
      grid_rf.fit(X_train, y_train)

      best_model_rf_tuned = grid_rf.best_estimator_
      y_pred_rf_tuned = best_model_rf_tuned.predict(X_test)

      acc_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
      prec_rf_tuned = precision_score(y_test, y_pred_rf_tuned, average='macro')
      rec_rf_tuned = recall_score(y_test, y_pred_rf_tuned, average='macro')
      f1_rf_tuned = f1_score(y_test, y_pred_rf_tuned, average='macro')

      mlflow.log_params(grid_rf.best_params_)
      mlflow.log_metric("accuracy", acc_rf_tuned)
      mlflow.log_metric("precision", prec_rf_tuned)
      mlflow.log_metric("recall", rec_rf_tuned)
      mlflow.log_metric("f1_score", f1_rf_tuned)

      mlflow.sklearn.log_model(best_model_rf_tuned, "best_model_rf_tuned_artifact")

      print("Random Forest (Tuned) - Best Params:", grid_rf.best_params_)
      print(f"Random Forest (Tuned) - Accuracy: {acc_rf_tuned:.4f} | F1: {f1_rf_tuned:.4f}")

      if f1_rf_tuned > best_f1:
          best_f1 = f1_rf_tuned
          best_model = best_model_rf_tuned
          best_run_id = run.info.run_id
          best_model_name = "RandomForest_Tuned"

  # --- Final comparison and registration of the overall best model ---
  if best_model is None:
      print("No model was successfully trained or identified as best.")
      return None

  print(f"\n--- Overall Best Model ---")
  print(f"Best Model Identified: {best_model_name}")
  print(f"Best F1 Score: {best_f1:.4f}")

  try:
      mlflow.set_experiment("Iris_Comparison") # Final comparison experiment
      with mlflow.start_run(run_name="Best_Model_Registration") as run:
          mlflow.log_metric("final_best_f1_score", best_f1)
          mlflow.log_param("best_model_type", best_model_name)
          # Log the best model found across all previous runs
          mlflow.sklearn.log_model(best_model, "final_best_iris_model",
                                  registered_model_name="IrisBestModel")
          print(f"Registered '{best_model_name}' as 'IrisBestModel' with F1: {best_f1:.4f}")
          return f"models:/IrisBestModel/latest"

  except Exception as e:
      print(f"Error during best model registration: {e}")
      return None


  if name == "main": 
    # Example usage when running this script directly for testing purposes. 
    # Set these paths according to your project structure. 
    # Assumes 'train_pipeline.py' is in 'notebooks/' and data is in 'Data/processed'. 
    BASE_DIR = os.path.dirname(os.path.abspath(file)) 
    PROCESSED_DATA_DIR_DEFAULT = os.path.join(BASE_DIR, '../Data/processed') 
    MODEL_DIR_DEFAULT = os.path.join(BASE_DIR, '../models')

    # Ensure dummy processed data exists for testing the function in isolation.
  # In a real retraining pipeline, a separate script would handle data pre-processing.
  os.makedirs(PROCESSED_DATA_DIR_DEFAULT, exist_ok=True)
  os.makedirs(MODEL_DIR_DEFAULT, exist_ok=True)

  train_data_npz = os.path.join(PROCESSED_DATA_DIR_DEFAULT, 'train_data.npz')
  test_data_npz = os.path.join(PROCESSED_DATA_DIR_DEFAULT, 'test_data.npz')
  feature_names_joblib = os.path.join(PROCESSED_DATA_DIR_DEFAULT, 'feature_names.joblib')
  scaler_joblib = os.path.join(PROCESSED_DATA_DIR_DEFAULT, 'scaler.joblib')

  if not os.path.exists(train_data_npz) or \
    not os.path.exists(test_data_npz) or \
    not os.path.exists(feature_names_joblib) or \
    not os.path.exists(scaler_joblib):
      print("Creating dummy processed data for demonstration...")
      from sklearn.datasets import load_iris
      from sklearn.preprocessing import StandardScaler
      from sklearn.model_selection import train_test_split

      iris = load_iris()
      X, y = iris.data, iris.target

      X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X, y, test_size=0.2, random_state=42)

      scaler_dummy = StandardScaler()
      X_train_scaled_dummy = scaler_dummy.fit_transform(X_train_dummy)
      X_test_scaled_dummy = scaler_dummy.transform(X_test_dummy)

      np.savez(train_data_npz, X=X_train_scaled_dummy, y=y_train_dummy)
      np.savez(test_data_npz, X=X_test_scaled_dummy, y=y_test_dummy)
      joblib.dump(iris.feature_names, feature_names_joblib)
      joblib.dump(scaler_dummy, scaler_joblib)
      print("Dummy processed data created.")

  print("\nStarting model training pipeline...")
  registered_model_uri = run_training_pipeline(PROCESSED_DATA_DIR_DEFAULT, MODEL_DIR_DEFAULT)

  if registered_model_uri:
      print(f"\nSuccessfully completed training pipeline.")
      print(f"Best model registered at: {registered_model_uri}")
  else:
      print("\nModel training pipeline failed.")
