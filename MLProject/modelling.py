import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ========================================================
# 1. INTEGRASI MLFLOW
# ========================================================
REPO_OWNER = "alfigrace"  
REPO_NAME = "submission-membangun-sistem-machine-learning"

mlflow.set_tracking_uri("file:./mlruns")

# Set nama eksperimen di MLflow Dashboard
mlflow.set_experiment("Heart_Disease_Detection")

mlflow.sklearn.autolog()

# ========================================================
# 2. LOAD DATASET HASIL PREPROCESSING
# ========================================================
DATA_DIR = "./heart_disease_preprocessing"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))

# Pisahkan Fitur (X) dan Target (y) - Kolom terakhir adalah target
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

X_val = val_df.iloc[:, :-1]
y_val = val_df.iloc[:, -1]

# ========================================================
# 3. PELATIHAN BASELINE MODEL & PENCATATAN MLFLOW
# ========================================================
params = {
    "n_estimators": 50,
    "max_depth": 5,
    "random_state": 42
}

print("🚀 Memulai eksperimen Pelatihan Baseline Model...")

with mlflow.start_run(run_name="Baseline_Random_Forest"):
    # Log Parameter ke MLflow
    mlflow.log_params(params)
    
    # Inisialisasi dan Latih Model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Prediksi Data Validasi
    y_pred = model.predict(X_val)
    
    # Hitung Metrik Evaluasi
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"Hasil Baseline -> Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")
    
    # Log Metrik ke MLflow
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    
    # Simpan Model sebagai Artifak di MLflow Cloud
    mlflow.sklearn.log_model(model, artifact_path="model_baseline")