
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
import os

def main():
    # File paths
    input_csv = '/mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast/all_merged_features.csv'
    output_zscore_csv = '/mnt/dataset4/cx/code/EEG_LLM_text/TUAB_fast/all_merged_features_zscored.csv'

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Identify target
    if 'primary_label' in df.columns:
        target_col = 'primary_label'
    elif 'labels' in df.columns:
        target_col = 'labels'
    else:
        raise ValueError("Could not find label column (primary_label or labels)")

    print(f"Target column: {target_col}")
    
    # Metadata columns to exclude
    exclude_cols = ['trial_ids', 'segment_ids', 'session_id', 'primary_label', 'labels', 
                    'start_time', 'end_time', 'total_time_length', 'merge_count', 
                    'source_segments', 'source_file']
    
    # Use ALL numeric columns as features
    feature_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"Selected {len(feature_cols)} features: {feature_cols}")

    # Drop rows with NaNs in feature columns or target
    df_clean = df.dropna(subset=feature_cols + [target_col]).copy()
    
    if len(df_clean) < len(df):
        print(f"Dropped {len(df) - len(df_clean)} rows containing NaN values.")
    
    # --- Part 1: Create Z-score CSV ---
    print("Generating Z-score normalized CSV...")
    
    # Create a copy for z-scored data
    df_zscored = df_clean.copy()
    scaler_all = StandardScaler()
    df_zscored[feature_cols] = scaler_all.fit_transform(df_clean[feature_cols])
    
    # Save Z-scored CSV
    df_zscored.to_csv(output_zscore_csv, index=False)
    print(f"Saved Z-score normalized data to {output_zscore_csv}")

    # --- Part 2: Train Models ---
    
    # X and y for Raw Data
    X_raw = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # X for Z-scored Data (using the same rows)
    X_zscored = df_zscored[feature_cols]
    
    # Split data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2, random_state=42)
    X_train_z, X_test_z, _, _ = train_test_split(X_zscored, y, test_size=0.2, random_state=42)
    
    print("\nTraining and Evaluating Models with ALL features...")
    
    results = []

    # Helper function to train and evaluate
    def evaluate_model(name, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return acc, f1

    # 1. Logistic Regression - Raw
    lr_raw = LogisticRegression(max_iter=2000, random_state=42) 
    acc_lr_raw, f1_lr_raw = evaluate_model("LR (Raw)", lr_raw, X_train_raw, X_test_raw, y_train, y_test)
    results.append(("Logistic Regression (Raw)", acc_lr_raw, f1_lr_raw))

    # 2. Logistic Regression - Z-score
    lr_z = LogisticRegression(max_iter=2000, random_state=42)
    acc_lr_z, f1_lr_z = evaluate_model("LR (Z-score)", lr_z, X_train_z, X_test_z, y_train, y_test)
    results.append(("Logistic Regression (Z-score)", acc_lr_z, f1_lr_z))

    # 3. XGBoost - Raw
    xgb_raw = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    acc_xgb_raw, f1_xgb_raw = evaluate_model("XGBoost (Raw)", xgb_raw, X_train_raw, X_test_raw, y_train, y_test)
    results.append(("XGBoost (Raw)", acc_xgb_raw, f1_xgb_raw))

    # 4. XGBoost - Z-score
    xgb_z = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    acc_xgb_z, f1_xgb_z = evaluate_model("XGBoost (Z-score)", xgb_z, X_train_z, X_test_z, y_train, y_test)
    results.append(("XGBoost (Z-score)", acc_xgb_z, f1_xgb_z))

    # Print Results Table
    print("\n" + "="*60)
    print(f"{'Model & Data':<35} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-" * 60)
    for name, acc, f1 in results:
        print(f"{name:<35} | {acc:.4f}     | {f1:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
