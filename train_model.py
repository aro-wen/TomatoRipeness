"""
Train an SVM classifier for tomato ripeness using the provided CSV.
Creates/overwrites:
 - svm_lab.joblib
 - label_encoder.joblib

Features: the app expects 6 features [L_mean, L_std, a_mean, a_std, b_mean, b_std].
The provided CSV contains L,a,b (single values). We'll expand each row to 6 features by
using the provided L,a,b and synthetic small std values (1.0) to match the app's input shape.

This script prints a quick train/test accuracy and classification report.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

CSV_PATH = "tomato_lab_synthetic_with_turning.csv"
MODEL_OUT = "svm_lab.joblib"
LE_OUT = "label_encoder.joblib"

if __name__ == '__main__':
    print("Loading CSV:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("Rows:", len(df))

    # Keep only desired labels and columns
    wanted_labels = ["ripe", "unripe", "turning", "rotten"]
    df = df[df['label'].isin(wanted_labels)].copy()
    print("Rows after filtering labels:", len(df))

    # Use L,a,b as means. Create synthetic std columns (small positive constant)
    df['L_mean'] = df['L'].astype(float)
    df['a_mean'] = df['a'].astype(float)
    df['b_mean'] = df['b'].astype(float)
    df['L_std'] = 1.0
    df['a_std'] = 1.0
    df['b_std'] = 1.0

    # Build feature matrix matching the app's expected order
    X = df[['L_mean','L_std','a_mean','a_std','b_mean','b_std']].to_numpy(dtype=np.float32)
    y = df['label'].astype(str).to_numpy()

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", list(le.classes_))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # Pipeline: scaler + SVM (RBF)
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, class_weight='balanced', random_state=42))

    print("Training SVM...")
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save artifacts
    print("Saving model to:", MODEL_OUT)
    joblib.dump(clf, MODEL_OUT)
    print("Saving label encoder to:", LE_OUT)
    joblib.dump(le, LE_OUT)

    print("Done.")
