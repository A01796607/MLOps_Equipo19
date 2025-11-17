import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
import random # New library for randomness

# --- MLOPS CONFIGURATION SIMULATION ---
MODELS_DIR = Path("models")
PROCESSED_DATA_DIR = Path("processed_data")
PIPELINE_PATH = MODELS_DIR / "random_forest_pipeline.pkl"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
TEST_LABELS_PATH = PROCESSED_DATA_DIR / "y_test_encoded.pkl"

# --- HELPER FUNCTIONS (Kept from previous version) ---

def create_preprocessing_pipeline(use_pca: bool = True, n_components: int = 50):
    steps = [('scaler', StandardScaler())]
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=42)))
    return Pipeline(steps)

def create_model_pipeline(preprocessor: Pipeline, model_type: str = 'random_forest', **model_params):
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=model_params.get('n_estimators', 200), random_state=42)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline([('preprocessor', preprocessor), ('model', model)])

def save_artifacts(rf_pipeline, label_encoder, X_test, y_test_encoded):
    MODELS_DIR.mkdir(exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    joblib.dump(rf_pipeline, PIPELINE_PATH)
    joblib.dump(label_encoder, PROCESSED_DATA_DIR / "label_encoder.pkl")
    joblib.dump(X_test, TEST_DATA_PATH)
    joblib.dump(y_test_encoded, TEST_LABELS_PATH)

def evaluate_model(y_true, y_pred, average='weighted'):
    return f1_score(y_true, y_pred, average=average)

def load_monitoring_artifacts():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        X_test = joblib.load(TEST_DATA_PATH)
        y_test_encoded = joblib.load(TEST_LABELS_PATH)
        return pipeline, X_test, y_test_encoded
    except FileNotFoundError:
        print("Error: Artifacts not found. Run initial training simulation first.")
        sys.exit(1)

# --- NEW: FUNCTION TO SIMULATE RANDOMIZED DRIFT ---

def simulate_data_drift_random(
    X_ref: pd.DataFrame, 
    shift_factor: float, 
    missing_ratio: float
) -> pd.DataFrame:
    """
    Generates a monitoring dataset with randomized drift parameters.
    """
    X_drift = X_ref.copy()
    
    # --- Type 1: Feature Drift (Shift in Mean) - Tempo ---
    tempo_feature = '_Tempo_Mean'
    original_mean = X_drift[tempo_feature].mean()
    # Apply the random shift (shift_factor > 1.0 means increase)
    delta = original_mean * (shift_factor - 1)
    X_drift[tempo_feature] = X_drift[tempo_feature] + delta
    
    # --- Type 2: Feature Missingness - RMS Energy ---
    rms_feature = '_RMSenergy_Mean'
    n_samples = len(X_drift)
    n_missing = int(n_samples * missing_ratio)
    
    # Apply the random missingness ratio
    missing_indices = np.random.choice(X_drift.index, n_missing, replace=False)
    X_drift.loc[missing_indices, rms_feature] = np.nan
    
    # Fill NaNs with feature mean to allow the pipeline to run
    X_drift.fillna(X_drift.mean(), inplace=True) 
    
    return X_drift, delta, n_missing

# --- NEW: FUNCTION TO RUN MULTIPLE EXPERIMENTS ---

def run_drift_experiment(
    pipeline: Pipeline, 
    X_ref: pd.DataFrame, 
    y_ref: np.ndarray, 
    threshold: float,
    num_iterations: int = 10
):
    """Runs the drift check multiple times with random parameters."""
    
    # Calculate and store baseline F1 (it's constant across runs)
    y_pred_ref = pipeline.predict(X_ref)
    baseline_f1 = evaluate_model(y_ref, y_pred_ref)
    
    print("\n" + "=" * 80)
    print(f"BASELINE F1-Weighted (Clean Data): {baseline_f1:.4f}")
    print(f"ALERT THRESHOLD: >{threshold:.2f}% DROP")
    print("=" * 80)
    print(f"Running {num_iterations} Randomized Drift Simulations...")
    print("-" * 80)
    
    all_results = []

    for i in range(1, num_iterations + 1):
        # 1. Randomize Drift Parameters for this run
        # Tempo Mean Shift: Randomly 10% to 30% increase
        shift_factor = random.uniform(1.10, 1.30)
        # Missingness: Randomly 5% to 20% of values are missing
        missing_ratio = random.uniform(0.05, 0.20)
        
        # 2. Simulate Drift
        X_drift, delta, n_missing = simulate_data_drift_random(
            X_ref, shift_factor, missing_ratio
        )
        
        # 3. Evaluate on Drifted Data
        y_pred_drift = pipeline.predict(X_drift)
        drift_f1 = evaluate_model(y_ref, y_pred_drift)

        # 4. Calculate Metrics
        performance_drop = baseline_f1 - drift_f1
        relative_drop_pct = (performance_drop / baseline_f1) * 100
        
        # 5. Log Results
        alert = "🚨 ALERT" if performance_drop > 0 and relative_drop_pct > threshold else "✅ OK"
        
        print(f"Run {i:02d}: F1={drift_f1:.4f} | Drop={relative_drop_pct:5.2f}% ({alert}) | Tempo $\Delta={delta:.2f}$ | Missing={n_missing}")

        all_results.append({
            'run': i,
            'drift_f1': drift_f1,
            'relative_drop_pct': relative_drop_pct,
            'alert': alert
        })

    # --- Final Summary Report ---
    results_df = pd.DataFrame(all_results)
    
    print("\n" + "=" * 80)
    print("Multi-Run Drift Simulation Summary")
    print("=" * 80)
    print(f"Total Runs: {num_iterations}")
    print(f"Runs with ALERT (Drop > {threshold:.2f}%): {results_df[results_df['alert'] == '🚨 ALERT'].shape[0]}")
    
    print("\nPerformance Drop (%) Statistics:")
    print(f" - Average Drop: {results_df['relative_drop_pct'].mean():.2f}%")
    print(f" - Worst-Case Drop (Max): {results_df['relative_drop_pct'].max():.2f}%")
    print(f" - Best-Case Drop (Min): {results_df['relative_drop_pct'].min():.2f}%")
    print("=" * 80)


def main_multi_drift_detection(data_path: str = "turkis_music_emotion_original.csv"):
    """Main execution function for multi-run drift detection."""
    
    # --- STEP 1: Simulating Initial Model Training (Run only once) ---
    print("--- STEP 1: Simulating Initial Model Training and Artifact Saving ---")
    df = pd.read_csv(data_path)
    target_col = "Class"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    preprocessor = create_preprocessing_pipeline(use_pca=True, n_components=50)
    rf_pipeline = create_model_pipeline(preprocessor, model_type='random_forest', n_estimators=200)
    rf_pipeline.fit(X_train, y_train_encoded)
    save_artifacts(rf_pipeline, label_encoder, X_test, y_test_encoded)
    
    # --- STEP 2: Running Multiple Drift Detection Scenarios ---
    
    pipeline, X_test_ref, y_test_ref_encoded = load_monitoring_artifacts()

    # Define the criteria for alert
    ALERT_THRESHOLD_PCT = 5.0

    # Run the core experiment
    run_drift_experiment(
        pipeline=pipeline, 
        X_ref=X_test_ref, 
        y_ref=y_test_ref_encoded, 
        threshold=ALERT_THRESHOLD_PCT,
        num_iterations=10 # Set the number of runs here
    )


if __name__ == "__main__":
    main_multi_drift_detection()