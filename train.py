import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import load_data, preprocess, check_data

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ── Config ──────────────────────────────────────────────
DATA_PATH   = 'data/heart.csv'
MODEL_PATH  = 'models/model.pkl'
SCALER_PATH = 'models/scaler.pkl'
RANDOM_STATE = 42

def train_and_evaluate():
    # 1. Load & inspect
    print("Loading data...")
    df = load_data(DATA_PATH)
    check_data(df)

    # 2. Preprocess
    X, y, scaler = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # 3. Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM':                 SVC(probability=True, random_state=RANDOM_STATE),
    }

    results = {}
    print("\n── Model Comparison ─────────────────────────────")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        cv     = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
        results[name] = {'accuracy': acc, 'auc': auc, 'cv_accuracy': cv}
        print(f"{name:25s} → Acc: {acc:.4f} | AUC: {auc:.4f} | CV: {cv:.4f}")

    # 4. Pick best model by AUC
    best_name  = max(results, key=lambda k: results[k]['auc'])
    best_model = models[best_name]
    print(f"\n✅ Best model: {best_name}")

    # 5. Detailed report for best model
    y_pred = best_model.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Disease']))

    # 6. Confusion matrix plot
    os.makedirs('models', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix — {best_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved → models/confusion_matrix.png")

    # 7. Feature importance (if Random Forest wins)
    if best_name == 'Random Forest':
        feat_imp = pd.Series(
            best_model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        plt.figure(figsize=(8, 5))
        feat_imp.plot(kind='bar', color='steelblue')
        plt.title('Feature Importance — Random Forest')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=150)
        plt.close()
        print("Feature importance saved → models/feature_importance.png")

    # 8. Save model & scaler
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    print(f"\nModel  saved → {MODEL_PATH}")
    print(f"Scaler saved → {SCALER_PATH}")

    return best_model, scaler, results

if __name__ == '__main__':
    train_and_evaluate()
