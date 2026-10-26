"""
train_ckd_model.py (v2)
Stage 2 - Chronic Kidney Disease (CKD) Risk Prediction
Cross-validated hyperparameter tuning + ROC-AUC based comparison.
Saves a confusion matrix + ROC curve image for the report/dashboard.
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

NUMERIC_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo"]
CATEGORICAL_COLS = ["htn", "dm"]
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS


def load_and_clean():
    df = pd.read_csv("data/ckd_raw.csv")
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str).str.strip().str.replace("\t", "", regex=False)
        df[col] = df[col].replace({"nan": np.nan, "?": np.nan, "": np.nan})

    df["classification"] = df["classification"].replace({"ckd\\t": "ckd"})
    df["target"] = df["classification"].map({"ckd": 1, "notckd": 0})

    numeric_like = NUMERIC_COLS + ["pcv", "wc", "rc"]
    for col in numeric_like:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0})
        df[col] = df[col].fillna(df[col].mode()[0])

    df = df.dropna(subset=["target"])
    keep = ALL_FEATURES + ["target"]
    df_clean = df[keep].reset_index(drop=True)
    df_clean.to_csv("data/ckd_clean.csv", index=False)
    return df_clean


def main():
    os.makedirs("static/img", exist_ok=True)
    df = load_and_clean()
    X = df[ALL_FEATURES]
    y = df["target"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    param_grids = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=42),
            {"C": [0.1, 0.5, 1, 2, 5]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"n_neighbors": [3, 5, 7, 9], "weights": ["distance"]},
        ),
        "SVM": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {"C": [0.5, 1, 2, 5], "gamma": ["scale", "auto"]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"n_estimators": [200, 400], "max_depth": [4, 6, 8],
             "min_samples_leaf": [1, 2]},
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
        search.fit(X_train_s, y_train)
        model = search.best_estimator_

        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)[:, 1]
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
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
        coefs = np.abs(best_model.coef_[0])
        coefs = coefs / coefs.sum()
        feature_importance = dict(zip(ALL_FEATURES, coefs.tolist()))

    preds_best = best_model.predict(X_test_s)
    proba_best = best_model.predict_proba(X_test_s)[:, 1]
    cm = confusion_matrix(y_test, preds_best)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(cm, cmap="Reds")
    axes[0].set_title(f"Confusion Matrix — {best_name}")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Actual")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Low Risk", "High Risk"])
    axes[0].set_yticklabels(["Low Risk", "High Risk"])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    fpr, tpr, _ = roc_curve(y_test, proba_best)
    axes[1].plot(fpr, tpr, color="#A9433D", linewidth=2, label=f"AUC = {best_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("static/img/ckd_eval.png", dpi=130)
    plt.close()

    joblib.dump(best_model, "models/ckd_model.pkl")
    joblib.dump(scaler, "models/ckd_scaler.pkl")

    with open("models/ckd_metrics.json", "w") as f:
        json.dump({
            "results": results,
            "best_model": best_name,
            "feature_importance": feature_importance,
            "feature_names": ALL_FEATURES,
        }, f, indent=2)

    print("Saved: models/ckd_model.pkl, models/ckd_scaler.pkl, models/ckd_metrics.json")
    print("Saved: static/img/ckd_eval.png")


if __name__ == "__main__":
    main()
