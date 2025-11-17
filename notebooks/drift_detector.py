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

# --- MLOPS CONFIGURATION SIMULATION ---
# These paths will be created relative to where you run the script
MODELS_DIR = Path("models")
PROCESSED_DATA_DIR = Path("processed_data")
MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

PIPELINE_PATH = MODELS_DIR / "random_forest_pipeline.pkl"
ENCODER_PATH = PROCESSED_DATA_DIR / "label_encoder.pkl"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "X_test.pkl"
TEST_LABELS_PATH = PROCESSED_DATA_DIR / "y_test_encoded.pkl"

# --- CORE FUNCTIONS (Based on your provided code) ---

def create_preprocessing_pipeline(use_pca: bool = True, n_components: int = 50):
    """Creates an sklearn Pipeline for preprocessing."""
    steps = [
        ('scaler', StandardScaler()),
    ]
    if use_pca:
        steps.append(('pca', PCA(n_components=n_components, random_state=42)))
    return Pipeline(steps)

def create_model_pipeline(preprocessor: Pipeline, model_type: str = 'random_forest', **model_params):
    """Creates a complete sklearn Pipeline with preprocessing and model."""
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=model_params.get('n_estimators', 200),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return full_pipeline

def save_artifacts(rf_pipeline, label_encoder, X_test, y_test_encoded):
    """Saves the trained pipeline, encoder, and test set for simulation."""
    joblib.dump(rf_pipeline, PIPELINE_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(X_test, TEST_DATA_PATH)
    joblib.dump(y_test_encoded, TEST_LABELS_PATH)
    # print(f"Artifacts saved to {MODELS_DIR} and {PROCESSED_DATA_DIR}")

def evaluate_model(y_true, y_pred, average='weighted'):
    """Calculates F1-Score (using weighted average)."""
    return f1_score(y_true, y_pred, average=average)

# --- DRIFT SIMULATION AND DETECTION LOGIC ---

def load_monitoring_artifacts():
    """Loads necessary artifacts for monitoring from disk."""
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        X_test = joblib.load(TEST_DATA_PATH)
        y_test_encoded = joblib.load(TEST_LABELS_PATH)
        label_encoder = joblib.load(ENCODER_PATH)
        return pipeline, X_test, y_test_encoded, label_encoder
    except FileNotFoundError:
        print("Error: Artifacts not found. Run initial training simulation first.")
        sys.exit(1)


def simulate_data_drift(X_ref: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a monitoring dataset (simulated production data) 
    with a change in distribution and missing values.
    """
    print("--- Simulating Data Drift in Monitoring Data ---")
    X_drift = X_ref.copy()

    # --- Type 1: Feature Drift (Shift in Mean) - Tempo ---
    # Simulates a 20% increase in average tempo
    tempo_feature = '_Tempo_Mean'
    original_mean = X_drift[tempo_feature].mean()
    delta = original_mean * (1.20 - 1)
    X_drift[tempo_feature] = X_drift[tempo_feature] + delta

    # --- Type 2: Feature Missingness - RMS Energy ---
    # Introduce 10% missing values
    rms_feature = '_RMSenergy_Mean'
    missing_ratio = 0.10
    n_samples = len(X_drift)
    n_missing = int(n_samples * missing_ratio)
    
    missing_indices = np.random.choice(X_drift.index, n_missing, replace=False)
    X_drift.loc[missing_indices, rms_feature] = np.nan
    
    # Fill NaNs with feature mean to allow the pipeline to run
    # (This demonstrates a simple/failing production imputation)
    X_drift.fillna(X_drift.mean(), inplace=True)
    
    print(f" - '{tempo_feature}' shifted by: +{delta:.2f}")
    print(f" - '{rms_feature}' introduced {n_missing} missing values (10%)")

    return X_drift

def run_performance_check(pipeline: Pipeline, X_ref: pd.DataFrame, y_ref: np.ndarray, threshold: float):
    """Evaluates the model on the reference data and the drifted data."""
    
    # 1. Evaluate on Reference Data (Establish Baseline)
    y_pred_ref = pipeline.predict(X_ref)
    baseline_f1 = evaluate_model(y_ref, y_pred_ref)
    
    # 2. Simulate and Evaluate on Drifted Data
    X_drift = simulate_data_drift(X_ref)
    y_pred_drift = pipeline.predict(X_drift)
    drift_f1 = evaluate_model(y_ref, y_pred_drift)

    # 3. Alerting Logic
    performance_drop = baseline_f1 - drift_f1
    relative_drop_pct = (performance_drop / baseline_f1) * 100

    print("\n" + "=" * 60)
    print("Data Drift Detection Report")
    print("=" * 60)
    print(f"BASELINE F1-Weighted (Clean Data): {baseline_f1:.4f}")
    print(f"PRODUCTION F1-Weighted (Drifted): {drift_f1:.4f}")
    print("-" * 60)
    
    alert_status = "✅ Status: OK (Performance drop is minimal)"
    proposed_action = "Continue Monitoring."

    if performance_drop > 0 and relative_drop_pct > threshold:
        alert_status = f"🚨 ALERT: Performance Drop Detected!"
        proposed_action = (
            "1. Review Feature Pipeline: Investigate the shift in '_Tempo_Mean' and the "
            "source of missing values in '_RMSenergy_Mean'.\n"
            "   2. Retrain Model: Collect new data reflecting the current music trends (tempo) "
            "and update the pipeline to handle missing values robustly."
        )

    print(alert_status)
    print(f" - Absolute Drop: {performance_drop:.4f}")
    print(f" - Relative Drop: {relative_drop_pct:.2f}%")
    print(f" - Alert Threshold: >{threshold:.2f}%")
    
    print("\n--- PROPOSED ACTION ---")
    print(proposed_action)
    print("=" * 60)
    
    return baseline_f1, drift_f1, relative_drop_pct


def main_drift_detection(data_path: str = "turkis_music_emotion_original.csv"):
    """Main execution function for drift detection."""
    
    print("--- STEP 1: Simulating Initial Model Training and Artifact Saving ---")
    df = pd.read_csv(data_path)
    target_col = "Class"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Split, Encode, Create Pipeline (Matching your initial script)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    preprocessor = create_preprocessing_pipeline(use_pca=True, n_components=50)
    rf_pipeline = create_model_pipeline(preprocessor, model_type='random_forest', n_estimators=200)
    rf_pipeline.fit(X_train, y_train_encoded)

    # Save artifacts for the monitoring step to load
    save_artifacts(rf_pipeline, label_encoder, X_test, y_test_encoded)
    
    # --- STEP 2: Data Drift Detection Phase ---
    print("\n--- STEP 2: Loading Model and Running Monitoring Check ---")
    
    pipeline, X_test_ref, y_test_ref_encoded, _ = load_monitoring_artifacts()

    # Define the criteria for alert: 5% drop in performance metric
    ALERT_THRESHOLD_PCT = 5.0

    # Run the core check
    run_performance_check(
        pipeline=pipeline, 
        X_ref=X_test_ref, 
        y_ref=y_test_ref_encoded, 
        threshold=ALERT_THRESHOLD_PCT
    )


if __name__ == "__main__":
    main_drift_detection()
