# Import required libraries
import os 
import sys 
import pandas as pd 
import numpy as np 
import shutil 
import joblib 
from datetime import datetime 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import load_iris 

# Get the directory of the current script
_script_dir = os.path.dirname(os.path.abspath(__file__))
# Assuming 'src' is a direct subdirectory of the project root, and this script is in the project root.
_project_root_dir = os.path.abspath(_script_dir)
# Add the 'src' directory to the beginning of the Python path
sys.path.insert(0, os.path.join(_project_root_dir, 'src'))

from src.data_utils import load_and_clean_data, prepare_features_and_target, split_and_scale_data

from train_pipeline import run_training_pipeline

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'raw') 
NEW_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'new') 
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'processed') 
ARCHIVED_DATA_DIR = os.path.join(PROJECT_ROOT, 'Data', 'archived') 
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')

IRIS_TARGET_COLUMN = 'target' # Iris dataset uses 'target' for species
ORIGINAL_IRIS_CSV_PATH = os.path.join(RAW_DATA_DIR, 'iris.csv') 
MIN_NEW_SAMPLES_FOR_RETRAIN = 5 # Minimum number of new samples to trigger retraining

# Ensure all necessary directories exist
os.makedirs(RAW_DATA_DIR, exist_ok=True) 
os.makedirs(NEW_DATA_DIR, exist_ok=True) 
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True) 
os.makedirs(ARCHIVED_DATA_DIR, exist_ok=True) 
os.makedirs(MODEL_DIR, exist_ok=True)

def create_initial_dummy_data(data_path): 
  """Creates a dummy iris.csv file if it doesn't exist.""" 
  if not os.path.exists(data_path): 
    print(f"Creating dummy initial data at {data_path} for demonstration.") 
    iris = load_iris(as_frame=True) # Load as DataFrame. This DataFrame has 'target' (numerical). 
    df = iris.frame
    # Ensure the 'target' column is named correctly (it usually is from load_iris)
    df.rename(columns={'target': IRIS_TARGET_COLUMN}, inplace=True)
    
    # IMPORTANT: Do NOT add the 'species' string column here.
    # The 'target' column is the numerical one used for training.
    # The 'species' string column can be added to new.csv for readability if desired,
    # but the pipeline will ignore/drop it if it's not the numerical target.
    
    df.to_csv(data_path, index=False)
    print(f"Iris dataset saved to {data_path}")
  else:
    print(f"Initial data already exists at {data_path}.")


# def create_initial_dummy_data(data_path): 
#   """Creates a dummy iris.csv file if it doesn't exist.""" 
#   if not os.path.exists(data_path): 
#     print(f"Creating dummy initial data at {data_path} for demonstration.") 
#     iris = load_iris(as_frame=True) # Load as DataFrame 
#     df = iris.frame # Rename target to 'target' as per the notebook's expectation 
#     df.rename(columns={'target': IRIS_TARGET_COLUMN}, inplace=True) # Ensure 'species' column is also present if the original notebook used it, # otherwise ensure 'target' is the numerical label. # The default load_iris(as_frame=True) has 'target' as numerical (0,1,2). # We also add 'species' for better readability of the raw data. 
#     df['species'] = iris.target_names[iris.target] 
#     df.to_csv(data_path, index=False) 
#     print(f"Iris dataset saved to {data_path}") 
#   else: print(f"Initial data already exists at {data_path}.")
  
def get_new_data_files(directory): 
  """Returns a list of full paths to new CSV files in the specified directory.""" 
  new_files = [f for f in os.listdir(directory) if f.endswith('.csv')] 
  return [os.path.join(directory, f) for f in new_files]

def combine_and_save_data(original_file, new_files, target_combined_file): 
  """ Combines original data with new data files, removes duplicates, and saves. Returns the combined DataFrame and the number of new unique samples added. Returns (None, 0) if a critical error occurs during combination. """ 
  df_original = pd.DataFrame() 
  initial_unique_rows_count = 0
  try:
    df_original = pd.read_csv(original_file)
    initial_unique_rows_count = len(df_original.drop_duplicates())
    print(f"Loaded original data with {len(df_original)} total rows ({initial_unique_rows_count} unique) from {original_file}")
  except FileNotFoundError:
      print(f"Warning: Original data file not found at {original_file}. Starting with new data only.")
  except Exception as e:
      print(f"Error loading original data file {original_file}: {e}")
      return None, 0

  new_data_frames = []
  for f_path in new_files:
      try:
          df_new = pd.read_csv(f_path)
          new_data_frames.append(df_new)
          print(f"Loaded new data file: {os.path.basename(f_path)} with {len(df_new)} rows.")
      except Exception as e:
          print(f"Error reading new data file {f_path}: {e}")
          continue

  if not new_data_frames and df_original.empty:
      print("No data (original or new) available to combine.")
      return pd.DataFrame(), 0

  try:
      df_combined = pd.concat([df_original] + new_data_frames, ignore_index=True, sort=False)
      df_combined.drop_duplicates(inplace=True)

      new_unique_samples = len(df_combined) - initial_unique_rows_count
      new_unique_samples = max(0, new_unique_samples) # Ensure not negative

      if new_unique_samples > 0:
          df_combined.to_csv(target_combined_file, index=False)
          print(f"Combined data saved to {target_combined_file} with {len(df_combined)} unique rows.")
      else:
          print("No new unique samples added to the combined dataset.")

      return df_combined, new_unique_samples

  except Exception as e:
      print(f"An unhandled error occurred during data combination or processing (concat/duplicates/save): {e}")
      return None, 0

def preprocess_and_save_data(df, processed_dir): 
  """ Preprocesses the combined DataFrame (feature engineering, cleaning, scaling, split) and saves the processed data and artifacts to the processed_dir. Returns True on success, False on failure. """ 
  print("Starting data preprocessing (including feature engineering)...")
  # --- Debug prints at the very beginning of the function ---
  print("\n--- DEBUG: Input DataFrame to preprocess_and_save_data ---")
  print(f"Shape: {df.shape}")
  print(f"Columns: {df.columns.tolist()}")
  print("Head:\n", df.head().to_string()) # Use to_string() to ensure full display
  print("Info:")
  df.info()
  print("-----------------------------------------------------------\n")

  # --- Feature Engineering (from Model_Training.ipynb) ---
  try:
      # Standardize column names first (e.g., 'sepal length (cm)' to 'sepal_length')
      df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('_(cm)', '', regex=False)

      print("\n--- DEBUG: DataFrame after initial column renaming ---")
      print(f"Columns after renaming: {df.columns.tolist()}")
      print("Head after renaming:\n", df.head().to_string())
      print("Info after renaming:")
      df.info()
      print("-----------------------------------------------------------\n")

      # Check for duplicate columns after renaming
      if len(df.columns) != len(set(df.columns)):
          print("CRITICAL ERROR: Duplicate column names detected after initial renaming!")
          print(f"Duplicate columns: {[col for col in df.columns if df.columns.tolist().count(col) > 1]}")
          return False

      # Ensure required columns for FE exist
      required_cols_for_fe = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
      if not all(col in df.columns for col in required_cols_for_fe):
          print(f"Error: Missing one or more required columns for feature engineering: {required_cols_for_fe}. Found: {df.columns.tolist()}")
          return False

      # Perform feature engineering
      # These operations will create new columns directly on `df`
      df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
      df["petal_area"] = df["petal_length"] * df["petal_width"]
      print("Feature engineering (SepalArea, PetalArea) completed.")

      print("\n--- DEBUG: DataFrame after feature engineering ---")
      print(f"Columns after FE: {df.columns.tolist()}")
      print("Head after FE:\n", df.head().to_string())
      print("Info after FE:")
      df.info()
      print("---------------------------------------------------\n")

  except Exception as e:
      print(f"An unexpected error occurred during feature engineering: {e}")
      return False

  # --- Data Cleaning (using src.data_utils.load_and_clean_data) ---
  # This function now takes the DataFrame directly
  try:
      df_cleaned = load_and_clean_data(df) # Pass the DataFrame `df` directly
      print("Data cleaning (via src.data_utils.load_and_clean_data) completed.")
  except Exception as e:
      print(f"Error during data cleaning (src.data_utils.load_and_clean_data): {e}")
      return False

  # Check if 'target' column exists after cleaning
  if IRIS_TARGET_COLUMN not in df_cleaned.columns:
      print(f"Error: Target column '{IRIS_TARGET_COLUMN}' not found after cleaning. Cannot proceed.")
      return False

  # --- Prepare Features and Target (using src.data_utils.prepare_features_and_target) ---
  try:
      X, y = prepare_features_and_target(df_cleaned, target_column=IRIS_TARGET_COLUMN)
      print("Features and target prepared (via src.data_utils.prepare_features_and_target).")
  except Exception as e:
      print(f"Error during feature/target preparation (src.data_utils.prepare_features_and_target): {e}")
      return False

  # --- Validation after Feature/Target Prep ---
  if X.empty or y.empty:
      print("Error: Features (X) or target (y) DataFrame/Series is empty after preparation. Cannot proceed.")
      return False

  # --- Split and Scale Data (using src.data_utils.split_and_scale_data) ---
  try:
      X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale_data(X, y, test_size=0.2, random_state=42)
      print("Data split and scaled (via src.data_utils.split_and_scale_data).")
  except Exception as e:
      print(f"Error during data splitting and scaling (src.data_utils.split_and_scale_data): {e}")
      return False

  # --- Final Validation before Saving ---
  if len(X_train_scaled) == 0 or len(X_test_scaled) == 0:
      print("Error: Train or test set is empty after splitting. Adjust data or split ratio.")
      return False

  # --- Save Processed Data and Artifacts ---
  try:
      # Save train and test data
      train_data_path = os.path.join(processed_dir, 'train_data.npz')
      test_data_path = os.path.join(processed_dir, 'test_data.npz')
      np.savez(train_data_path, X=X_train_scaled, y=y_train)
      np.savez(test_data_path, X=X_test_scaled, y=y_test)

      # Save the scaler for later use
      scaler_path = os.path.join(processed_dir, 'scaler.joblib')
      joblib.dump(scaler, scaler_path)

      # Save feature names for reference
      # X.columns.tolist() will include the newly engineered features too
      feature_names_path = os.path.join(processed_dir, 'feature_names.joblib')
      joblib.dump(X.columns.tolist(), feature_names_path)

      # Also save the raw processed dataframe (df_cleaned with engineered features) for reference
      processed_data_path = os.path.join(processed_dir, 'processed_iris.csv')
      df_cleaned.to_csv(processed_data_path, index=False)

      print(f"Saved processed training data to {train_data_path}")
      print(f"Saved processed testing data to {test_data_path}")
      print(f"Saved scaler to {scaler_path}")
      print(f"Saved feature names to {feature_names_path}")
      print(f"Saved processed dataframe to {processed_data_path}")
      return True
  except Exception as e:
      print(f"Error saving processed data artifacts: {e}")
      return False

def move_files_to_archive(files, destination_dir): 
  """Moves files from source to destination directory.""" 
  for f_path in files: 
    try: 
      shutil.move(f_path, os.path.join(destination_dir, os.path.basename(f_path))) 
      print(f"Archived: {os.path.basename(f_path)}") 
    except Exception as e: 
      print(f"Error archiving file {f_path}: {e}")
      
def run_retraining_trigger(): 
  """ Main function to check for new data and trigger the retraining pipeline. """ 
  print(f"\n--- Model Retraining Trigger: {datetime.now()} ---")
  # 1. Ensure initial raw data exists for a starting point
  create_initial_dummy_data(ORIGINAL_IRIS_CSV_PATH)

  # 2. Check for new data files
  new_data_csv_files = get_new_data_files(NEW_DATA_DIR)

  if not new_data_csv_files:
      print("No new data files found in the 'new' directory. Skipping retraining.")
      return

  print(f"Found {len(new_data_csv_files)} new data files for processing.")

  # 3. Combine original and new data
  temp_combined_raw_data_path = os.path.join(RAW_DATA_DIR, 'combined_iris_data.csv')
  df_combined, new_unique_samples = combine_and_save_data(ORIGINAL_IRIS_CSV_PATH, new_data_csv_files, temp_combined_raw_data_path)

  # Check if df_combined is None, indicating an error in combine_and_save_data
  if df_combined is None:
      print("Data combination or initial processing failed. Cannot proceed with retraining.")
      move_files_to_archive(new_data_csv_files, ARCHIVED_DATA_DIR)
      return

  if df_combined.empty:
      print("Combined dataset is empty after processing. Skipping retraining.")
      move_files_to_archive(new_data_csv_files, ARCHIVED_DATA_DIR)
      return

  if new_unique_samples < MIN_NEW_SAMPLES_FOR_RETRAIN:
      print(f"Only {new_unique_samples} new unique samples added. Minimum required for retraining is {MIN_NEW_SAMPLES_FOR_RETRAIN}. Skipping retraining.")
      move_files_to_archive(new_data_csv_files, ARCHIVED_DATA_DIR)
      # The combined raw data file might still be updated in raw/, this is intended
      # as it represents the current full dataset.
      return

  # 4. Preprocess the combined data and save to PROCESSED_DATA_DIR
  # Pass the DataFrame directly to preprocess_and_save_data
  if not preprocess_and_save_data(df_combined, PROCESSED_DATA_DIR):
      print("Data preprocessing failed. Skipping retraining.")
      move_files_to_archive(new_data_csv_files, ARCHIVED_DATA_DIR)
      return

  # 5. Trigger the training pipeline
  print("\nTriggering the encapsulated model training pipeline...")
  registered_model_uri = run_training_pipeline(PROCESSED_DATA_DIR, MODEL_DIR)

  if registered_model_uri:
      print(f"\nSuccessfully completed model retraining.")
      print(f"New best model registered at: {registered_model_uri}")
      # Update the original data file with the newly combined data
      shutil.copy(temp_combined_raw_data_path, ORIGINAL_IRIS_CSV_PATH)
      print(f"Updated {os.path.basename(ORIGINAL_IRIS_CSV_PATH)} with the latest combined data.")
      # 6. Archive the new data files
      move_files_to_archive(new_data_csv_files, ARCHIVED_DATA_DIR)
  else:
      print("\nModel retraining pipeline failed. New data files were NOT archived (unless preprocessing failed).")

  # Clean up temporary combined raw data file
  if os.path.exists(temp_combined_raw_data_path):
      os.remove(temp_combined_raw_data_path)
      print(f"Cleaned up temporary file: {temp_combined_raw_data_path}")

if __name__ == "__main__":
  run_retraining_trigger()
