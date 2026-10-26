"""
explainability.py
Real SHAP (SHapley Additive exPlanations) based per-prediction factor breakdown.

SHAP computes, for a single prediction, how much each feature pushed the
model's output up or down relative to a baseline - a well-established,
published technique (Lundberg & Lee, 2017), not a custom heuristic.

Falls back to a simpler importance x deviation heuristic for model types
without a fast, exact SHAP explainer (e.g. KNN, or if SHAP raises for any
reason) so a prediction can never fail just because explanation did.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _heuristic_factors(feature_importance, feature_names, scaled_values, display_names, top_n):
    if not feature_importance:
        return []
    contributions = []
    for i, fname in enumerate(feature_names):
        importance = feature_importance.get(fname, 0)
        deviation = scaled_values[i]
        score = importance * abs(deviation)
        direction = "above average" if deviation > 0 else "below average"
        contributions.append({"name": display_names.get(fname, fname), "score": score, "direction": direction})
    contributions.sort(key=lambda c: c["score"], reverse=True)
    top = contributions[:top_n]
    total = sum(c["score"] for c in contributions) or 1
    for c in top:
        c["pct"] = round((c["score"] / total) * 100, 1)
    return top


def build_explainer(model, background_scaled):
    """Build the right SHAP explainer for this model type, once at app startup."""
    if not SHAP_AVAILABLE:
        return (None, None)
    try:
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            return ("tree", shap.TreeExplainer(model))
        if isinstance(model, LogisticRegression):
            return ("linear", shap.LinearExplainer(model, background_scaled))
    except Exception:
        pass
    return (None, None)


def compute_factors(explainer_kind, explainer, feature_names, scaled_row, display_names,
                     feature_importance=None, top_n=3):
    """Returns the top contributing factors for one prediction, with a method label
    so the UI can be transparent about whether real SHAP values or the fallback
    heuristic were used."""
    if explainer is not None:
        try:
            row = np.asarray(scaled_row).reshape(1, -1)
            if explainer_kind == "tree":
                raw = explainer.shap_values(row)
                if isinstance(raw, list):
                    vals = np.asarray(raw[1][0])
                elif np.ndim(raw) == 3:
                    vals = np.asarray(raw)[0, :, 1]
                else:
                    vals = np.asarray(raw)[0]
            else:  # linear
                vals = np.asarray(explainer.shap_values(row))[0]

            contributions = []
            for i, fname in enumerate(feature_names):
                score = abs(float(vals[i]))
                direction = "above average" if scaled_row[i] > 0 else "below average"
                contributions.append({"name": display_names.get(fname, fname), "score": score, "direction": direction})
            contributions.sort(key=lambda c: c["score"], reverse=True)
            top = contributions[:top_n]
            total = sum(c["score"] for c in contributions) or 1
            for c in top:
                c["pct"] = round((c["score"] / total) * 100, 1)
            return top, "shap"
        except Exception:
            pass  # fall through to heuristic on any SHAP failure

    return _heuristic_factors(feature_importance, feature_names, scaled_row, display_names, top_n), "heuristic"
