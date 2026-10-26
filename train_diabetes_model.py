"""
train_diabetes_model.py (v2)
Stage 1 - Diabetes Prediction
Adds engineered features, SMOTE class balancing, cross-validated
hyperparameter tuning, and ROC-AUC based model comparison.
Also saves a confusion matrix + ROC curve image for the report/dashboard.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE

COLUMNS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
           "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


def engineer_features(df):
    """Derive extra clinically-meaningful features from the 8 base inputs."""
    df = df.copy()
    df["Glucose_BMI"] = (df["Glucose"] * df["BMI"]) / 100.0
    df["Insulin_Glucose_Ratio"] = df["Insulin"] / (df["Glucose"] + 1)
    df["Age_Pregnancies"] = df["Age"] * (df["Pregnancies"] + 1) / 10.0

    df["BMI_Category"] = pd.cut(df["BMI"], bins=[0, 18.5, 25, 30, 100],
                                 labels=[0, 1, 2, 3]).astype(float)
    df["Glucose_Category"] = pd.cut(df["Glucose"], bins=[0, 100, 126, 400],
                                     labels=[0, 1, 2]).astype(float)
    return df


ENGINEERED_COLS = ["Glucose_BMI", "Insulin_Glucose_Ratio", "Age_Pregnancies",
                    "BMI_Category", "Glucose_Category"]
BASE_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
             "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
ALL_FEATURES = BASE_COLS + ENGINEERED_COLS


def load_and_clean():
    df = pd.read_csv("data/pima_raw.csv", header=None, names=COLUMNS)
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_invalid:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    df = engineer_features(df)
    df.to_csv("data/diabetes_clean.csv", index=False)
    return df


def main():
    os.makedirs("static/img", exist_ok=True)
    df = load_and_clean()
    X = df[ALL_FEATURES]
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_s, y_train)

    # ---- SMOTE before/after class balance visualization ----
    before_counts = y_train.value_counts().sort_index()
    after_counts = y_train_bal.value_counts().sort_index()
    fig_s, axes_s = plt.subplots(1, 2, figsize=(9, 3.6))
    axes_s[0].bar(["Not Diabetic", "Diabetic"], before_counts.values, color=["#5E9BD8", "#C8483F"])
    axes_s[0].set_title("Before SMOTE (Training Set)")
    axes_s[0].set_ylabel("Count")
    for i, v in enumerate(before_counts.values):
        axes_s[0].text(i, v + 2, str(v), ha="center", fontsize=9)
    axes_s[1].bar(["Not Diabetic", "Diabetic"], after_counts.values, color=["#5E9BD8", "#C8483F"])
    axes_s[1].set_title("After SMOTE (Training Set)")
    for i, v in enumerate(after_counts.values):
        axes_s[1].text(i, v + 2, str(v), ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("static/img/smote_balance.png", dpi=130)
    plt.close()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.1, 0.5, 1, 2, 5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [7, 9, 11, 13, 15], "weights": ["distance"]},
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {"C": [0.5, 1, 2, 5], "gamma": ["scale", "auto"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [200, 400], "max_depth": [5, 7, 9],
             "min_samples_leaf": [1, 2, 4]},
        ),
        "XGBoost": (
            XGBClassifier(random_state=42, eval_metric="logloss"),
            {"n_estimators": [150, 300], "max_depth": [3, 4, 5],
             "learning_rate": [0.05, 0.1]},
        ),
    }

    results = {}
    best_name, best_model, best_auc, best_f1 = None, None, -1, -1

    print(f"{'Model':<22}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>10}{'ROC-AUC':>10}")
    print("-" * 74)
    for name, (estimator, grid) in param_grids.items():
        search = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train_bal, y_train_bal)
        model = search.best_estimator_

        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, proba)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
                          "roc_auc": auc, "best_params": search.best_params_}
        print(f"{name:<22}{acc*100:>9.2f}%{prec*100:>11.2f}%{rec*100:>9.2f}%{f1*100:>9.2f}%{auc*100:>9.2f}%")

        if auc > best_auc or (auc == best_auc and f1 > best_f1):
            best_name, best_model, best_auc, best_f1 = name, model, auc, f1

    print("-" * 74)
    print(f"Best model: {best_name} (ROC-AUC={best_auc*100:.2f}%)")

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = dict(zip(ALL_FEATURES, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        # Linear models (e.g. Logistic Regression) expose coefficients instead;
        # use absolute, normalized coefficients as a feature-importance proxy.
        coefs = np.abs(best_model.coef_[0])
        coefs = coefs / coefs.sum()
        feature_importance = dict(zip(ALL_FEATURES, coefs.tolist()))

    preds_best = best_model.predict(X_test_s)
    proba_best = best_model.predict_proba(X_test_s)[:, 1]
    cm = confusion_matrix(y_test, preds_best)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix — {best_name}")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Not Diabetic", "Diabetic"])
    axes[0].set_yticklabels(["Not Diabetic", "Diabetic"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    fpr, tpr, _ = roc_curve(y_test, proba_best)
    axes[1].plot(fpr, tpr, color="#2E75B6", linewidth=2, label=f"AUC = {best_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("static/img/diabetes_eval.png", dpi=130)
    plt.close()

    joblib.dump(best_model, "models/diabetes_model.pkl")
    joblib.dump(scaler, "models/diabetes_scaler.pkl")

    with open("models/diabetes_metrics.json", "w") as f:
        json.dump({
            "results": results,
            "best_model": best_name,
            "feature_importance": feature_importance,
            "feature_names": ALL_FEATURES,
        }, f, indent=2)

    print("Saved: models/diabetes_model.pkl, models/diabetes_scaler.pkl, models/diabetes_metrics.json")
    print("Saved: static/img/diabetes_eval.png")


if __name__ == "__main__":
    main()
