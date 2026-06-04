import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_preprocessing(input_path, output_dir):
    print(f"Mengunduh data dari: {input_path}")
    
    # 1. Load Data
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} tidak ditemukan!")
        sys.exit(1)
        
    df = pd.read_csv(input_path)
    
    # 2. Hapus Duplikat
    df = df.drop_duplicates()
    
    # Separasi Fitur & Target
    X = df.drop(columns=['Heart Disease Status'])
    y = df['Heart Disease Status'].map({'Yes': 1, 'No': 0})
    
    # 3. Split Data (80:10:10)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 4. Handling Missing Values berbasis data Train
    kolom_angka = X_train.select_dtypes(include=['float64', 'int64']).columns
    kolom_kategori = X_train.select_dtypes(include=['object']).columns
    
    for col in kolom_angka:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_val[col] = X_val[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)
        
    for col in kolom_kategori:
        modus_val = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(modus_val)
        X_val[col] = X_val[col].fillna(modus_val)
        X_test[col] = X_test[col].fillna(modus_val)
        
    # 5. Penanganan Outlier (Capping IQR)
    for col in kolom_angka:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_val[col] = np.clip(X_val[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)
        
    # 6. Standarisasi Fitur
    scaler = StandardScaler()
    X_train[kolom_angka] = scaler.fit_transform(X_train[kolom_angka])
    X_val[kolom_angka] = scaler.transform(X_val[kolom_angka])
    X_test[kolom_angka] = scaler.transform(X_test[kolom_angka])
    
    # 7. Encoding Data Kategorikal
    X_train = pd.get_dummies(X_train, columns=kolom_kategori, drop_first=True)
    X_val = pd.get_dummies(X_val, columns=kolom_kategori, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=kolom_kategori, drop_first=True)
    
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # 8. Simpan Hasil
    os.makedirs(output_dir, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(f"{output_dir}/train.csv", index=False)
    pd.concat([X_val, y_val], axis=1).to_csv(f"{output_dir}/val.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f"{output_dir}/test.csv", index=False)
    
    print(f"✅ Otomatisasi Sukses! File tersimpan di folder: {output_dir}")

if __name__ == "__main__":
    # Menangkap input path dari terminal/command line
    input_data = "heart_disease_raw/heart_disease.csv"
    output_folder = "heart_disease_preprocessing"
    run_preprocessing(input_data, output_folder)