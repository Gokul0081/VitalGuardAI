"""
app.py
Diabetes Detection and Kidney Disease Risk Prediction using Machine Learning
Main Flask application.
"""
import os
import io
import re
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, PageBreak, HRFlowable)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER

import db
import explainability as xai
from chatbot import answer_question, SUGGESTED_QUESTIONS

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# ---------------------------------------------------------------
# Load trained models once at startup
# ---------------------------------------------------------------
diabetes_model = joblib.load(os.path.join(MODELS_DIR, "diabetes_model.pkl"))
diabetes_scaler = joblib.load(os.path.join(MODELS_DIR, "diabetes_scaler.pkl"))
with open(os.path.join(MODELS_DIR, "diabetes_metrics.json")) as f:
    diabetes_metrics = json.load(f)

ckd_model = joblib.load(os.path.join(MODELS_DIR, "ckd_model.pkl"))
ckd_scaler = joblib.load(os.path.join(MODELS_DIR, "ckd_scaler.pkl"))
with open(os.path.join(MODELS_DIR, "ckd_metrics.json")) as f:
    ckd_metrics = json.load(f)

db.init_db()

# ---------------------------------------------------------------
# SHAP explainers - built once at startup using a sample of real
# (cleaned) training data as the background distribution
# ---------------------------------------------------------------
def _load_background(csv_path, feature_names, scaler, n=100):
    df = pd.read_csv(csv_path)[feature_names]
    sample = df.sample(n=min(n, len(df)), random_state=42)
    return scaler.transform(sample)

try:
    _diabetes_bg = _load_background(os.path.join(BASE_DIR, "data", "diabetes_clean.csv"),
                                      diabetes_metrics["feature_names"], diabetes_scaler)
    diabetes_explainer_kind, diabetes_explainer = xai.build_explainer(diabetes_model, _diabetes_bg)
except Exception:
    diabetes_explainer_kind, diabetes_explainer = None, None

try:
    _ckd_bg = _load_background(os.path.join(BASE_DIR, "data", "ckd_clean.csv"),
                                 ckd_metrics["feature_names"], ckd_scaler)
    ckd_explainer_kind, ckd_explainer = xai.build_explainer(ckd_model, _ckd_bg)
except Exception:
    ckd_explainer_kind, ckd_explainer = None, None

BASE_DIABETES_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                       "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
CKD_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "htn", "dm"]

# ---------------------------------------------------------------
# Precaution / recommendation content (used on result pages + PDF report)
# ---------------------------------------------------------------
PRECAUTIONS = {
    "diabetic": [
        "Monitor your blood glucose levels regularly, ideally with a home glucometer.",
        "Follow a balanced, low-sugar, low-refined-carbohydrate diet; consult a dietitian if possible.",
        "Aim for at least 30 minutes of moderate physical activity most days of the week.",
        "Maintain a healthy body weight (target BMI range: 18.5 - 24.9 kg/m\u00b2).",
        "Avoid smoking and limit alcohol consumption.",
        "Schedule regular check-ups with an endocrinologist or physician.",
        "Get an HbA1c test done periodically to track long-term blood sugar control.",
        "Watch for warning symptoms: excessive thirst, frequent urination, unexplained fatigue, or blurred vision.",
    ],
    "not_diabetic": [
        "Maintain a balanced diet low in refined sugars and processed foods.",
        "Exercise regularly to support healthy insulin sensitivity.",
        "If you have a family history of diabetes, get screened periodically regardless of current results.",
        "Maintain a healthy BMI and blood pressure range.",
        "Limit sugary beverages and maintain good sleep hygiene.",
    ],
    "ckd_high_risk": [
        "Consult a nephrologist promptly for confirmatory tests (eGFR, urine albumin-to-creatinine ratio).",
        "Reduce salt and dietary protein intake as advised by your doctor.",
        "Keep blood pressure and blood sugar tightly controlled \u2014 both accelerate kidney damage.",
        "Stay adequately hydrated, but follow any fluid-restriction guidance from your physician.",
        "Avoid NSAIDs (e.g. ibuprofen) and other nephrotoxic medications unless prescribed.",
        "Get kidney function tests (serum creatinine, eGFR) done regularly to track progression.",
        "Avoid excessive potassium-rich foods if advised by your doctor (e.g. bananas, oranges) until evaluated.",
    ],
    "ckd_low_risk": [
        "Maintain good hydration and a balanced, kidney-friendly diet.",
        "Keep blood pressure and blood sugar within healthy ranges.",
        "Avoid overuse of over-the-counter painkillers (NSAIDs).",
        "Get periodic kidney function screening, especially if you are diabetic or hypertensive.",
        "Maintain a healthy, active lifestyle to support long-term kidney health.",
    ],
}

# ---------------------------------------------------------------
# Urgent warning signs — shown as a distinct, high-alert callout
# ---------------------------------------------------------------
URGENT_SIGNS = {
    "diabetic": [
        "Severe thirst or dry mouth combined with confusion or extreme drowsiness",
        "Rapid, deep breathing or breath with a fruity/acetone smell",
        "Persistent vomiting, severe abdominal pain, or very high blood sugar readings",
        "Loss of consciousness or seizures",
    ],
    "ckd_high_risk": [
        "Little to no urination, or swelling in the legs, ankles, or around the eyes",
        "Severe shortness of breath or chest pain",
        "Nausea, vomiting, or confusion that comes on suddenly",
        "Very high blood pressure readings alongside any of the above",
    ],
}

# ---------------------------------------------------------------
# Sample diet guidance — general, safe, non-prescriptive examples
# ---------------------------------------------------------------
DIET_GUIDANCE = {
    "diabetic": {
        "title": "Sample Diabetic-Friendly Day of Eating",
        "meals": [
            ("Breakfast", "Vegetable oats or whole-grain toast with eggs; unsweetened tea/coffee"),
            ("Lunch", "Brown rice or whole-wheat roti, dal, a non-starchy vegetable, and a small portion of lean protein"),
            ("Snack", "A handful of nuts or a small fruit (apple, guava) rather than juice"),
            ("Dinner", "Grilled fish/chicken/paneer with sauteed vegetables and a small portion of whole grains"),
        ],
        "avoid": "Sugary drinks, white bread/rice in large portions, deep-fried snacks, and sweetened desserts.",
    },
    "ckd_high_risk": {
        "title": "Sample Kidney-Friendly Day of Eating",
        "meals": [
            ("Breakfast", "Poha or upma with vegetables, limited salt"),
            ("Lunch", "Rice with a moderate-protein curry, avoiding high-potassium vegetables if advised"),
            ("Snack", "Apple or a small portion of low-potassium fruit"),
            ("Dinner", "Light vegetable soup with controlled portions of protein"),
        ],
        "avoid": "Excess salt, processed/canned foods, and high-potassium foods (bananas, oranges, potatoes) unless cleared by your doctor.",
    },
}

SPECIALIST_SEARCH = {
    "diabetic": ("Find an Endocrinologist Near You", "https://www.google.com/maps/search/endocrinologist+near+me"),
    "ckd_high_risk": ("Find a Nephrologist Near You", "https://www.google.com/maps/search/nephrologist+near+me"),
}


DIABETES_DISPLAY_NAMES = {
    "Pregnancies": "Pregnancies", "Glucose": "Glucose Level", "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness", "Insulin": "Insulin Level", "BMI": "BMI",
    "DiabetesPedigreeFunction": "Family History Score", "Age": "Age",
    "Glucose_BMI": "Glucose \u00d7 BMI Interaction", "Insulin_Glucose_Ratio": "Insulin/Glucose Ratio",
    "Age_Pregnancies": "Age/Pregnancy Combination", "BMI_Category": "BMI Category",
    "Glucose_Category": "Glucose Category",
}
CKD_DISPLAY_NAMES = {
    "age": "Age", "bp": "Blood Pressure", "sg": "Specific Gravity", "al": "Albumin",
    "su": "Sugar", "bgr": "Blood Glucose", "bu": "Blood Urea", "sc": "Serum Creatinine",
    "sod": "Sodium", "pot": "Potassium", "hemo": "Hemoglobin", "htn": "Hypertension", "dm": "Diabetes Mellitus",
}


def get_session_id():
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4())
        db.ensure_session(session["session_id"])
    return session["session_id"]


def get_current_user_id():
    return session.get("user_id")


def get_current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    return db.get_user_by_id(uid)


def admin_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("is_admin"):
            return redirect(url_for("admin_login"))
        return view_func(*args, **kwargs)
    return wrapped


def login_required(view_func):
    @wraps(view_func)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to access this page.", "danger")
            return redirect(url_for("login", next=request.path))
        return view_func(*args, **kwargs)
    return wrapped


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
SECURITY_QUESTIONS = [
    "What was your first pet's name?",
    "What is your mother's maiden name?",
    "What city were you born in?",
    "What was the name of your first school?",
]


def build_diabetes_features(form_values):
    """Compute the same engineered features used at training time."""
    base = pd.DataFrame([[
        form_values["pregnancies"], form_values["glucose"], form_values["blood_pressure"],
        form_values["skin_thickness"], form_values["insulin"], form_values["bmi"],
        form_values["pedigree"], form_values["age"],
    ]], columns=BASE_DIABETES_COLS)

    base["Glucose_BMI"] = (base["Glucose"] * base["BMI"]) / 100.0
    base["Insulin_Glucose_Ratio"] = base["Insulin"] / (base["Glucose"] + 1)
    base["Age_Pregnancies"] = base["Age"] * (base["Pregnancies"] + 1) / 10.0

    def bmi_cat(v):
        if v < 18.5: return 0
        if v < 25: return 1
        if v < 30: return 2
        return 3

    def glu_cat(v):
        if v < 100: return 0
        if v < 126: return 1
        return 2

    base["BMI_Category"] = base["BMI"].apply(bmi_cat).astype(float)
    base["Glucose_Category"] = base["Glucose"].apply(glu_cat).astype(float)

    return base[diabetes_metrics["feature_names"]]


# ---------------------------------------------------------------
# Public / User routes
# ---------------------------------------------------------------
@app.context_processor
def inject_user():
    return {"current_user": get_current_user()}


# ---------------------------------------------------------------
# Auth routes (optional accounts - guests keep working unchanged)
# ---------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("profile"))
    if request.method == "GET":
        return render_template("register.html", security_questions=SECURITY_QUESTIONS)

    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm = request.form.get("confirm_password", "")
    security_question = request.form.get("security_question", "")
    security_answer = request.form.get("security_answer", "").strip()

    errors = []
    if len(username) < 3:
        errors.append("Username must be at least 3 characters.")
    if not EMAIL_RE.match(email):
        errors.append("Please enter a valid email address.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    if security_question not in SECURITY_QUESTIONS:
        errors.append("Please select a security question.")
    if len(security_answer) < 2:
        errors.append("Please provide a security answer.")
    if db.get_user_by_username(username):
        errors.append("That username is already taken.")
    if db.get_user_by_email(email):
        errors.append("An account with that email already exists.")

    if errors:
        for e in errors:
            flash(e, "danger")
        return redirect(url_for("register"))

    user_id = db.create_user(
        username, email, generate_password_hash(password),
        security_question, generate_password_hash(security_answer.lower()),
    )
    if not user_id:
        flash("Could not create account. Please try a different username or email.", "danger")
        return redirect(url_for("register"))

    session["user_id"] = user_id
    session["username"] = username
    flash("Account created! You're now logged in.", "success")
    return redirect(url_for("profile"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("profile"))
    if request.method == "GET":
        return render_template("login.html", next_url=request.args.get("next", ""))

    identifier = request.form.get("identifier", "").strip()
    password = request.form.get("password", "")
    next_url = request.form.get("next_url") or url_for("profile")

    user = db.get_user_by_username(identifier) or db.get_user_by_email(identifier.lower())
    if user and check_password_hash(user["password_hash"], password):
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        return redirect(next_url)

    flash("Incorrect username/email or password.", "danger")
    return redirect(url_for("login"))


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("username", None)
    return redirect(url_for("index"))


@app.route("/profile")
@login_required
def profile():
    user = get_current_user()
    stats = db.get_user_stats(user["id"])
    return render_template("profile.html", user=user, stats=stats)


@app.route("/profile/change-password", methods=["POST"])
@login_required
def change_password():
    user = get_current_user()
    current_pw = request.form.get("current_password", "")
    new_pw = request.form.get("new_password", "")
    confirm_pw = request.form.get("confirm_new_password", "")

    if not check_password_hash(user["password_hash"], current_pw):
        flash("Current password is incorrect.", "danger")
    elif len(new_pw) < 8:
        flash("New password must be at least 8 characters.", "danger")
    elif new_pw != confirm_pw:
        flash("New passwords do not match.", "danger")
    else:
        db.update_password(user["id"], generate_password_hash(new_pw))
        flash("Password updated successfully.", "success")
    return redirect(url_for("profile"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "GET":
        return render_template("forgot_password.html", step="find")

    step = request.form.get("step", "find")

    if step == "find":
        identifier = request.form.get("identifier", "").strip()
        user = db.get_user_by_username(identifier) or db.get_user_by_email(identifier.lower())
        if not user:
            flash("No account found with that username or email.", "danger")
            return redirect(url_for("forgot_password"))
        return render_template("forgot_password.html", step="answer", username=user["username"],
                                question=user["security_question"])

    if step == "answer":
        username = request.form.get("username", "")
        answer = request.form.get("security_answer", "").strip().lower()
        user = db.get_user_by_username(username)
        if not user or not check_password_hash(user["security_answer_hash"], answer):
            flash("That answer doesn't match our records.", "danger")
            return render_template("forgot_password.html", step="answer", username=username,
                                    question=user["security_question"] if user else "")
        return render_template("forgot_password.html", step="reset", username=username)

    if step == "reset":
        username = request.form.get("username", "")
        new_pw = request.form.get("new_password", "")
        confirm_pw = request.form.get("confirm_new_password", "")
        user = db.get_user_by_username(username)
        if not user:
            flash("Something went wrong. Please start over.", "danger")
            return redirect(url_for("forgot_password"))
        if len(new_pw) < 8 or new_pw != confirm_pw:
            flash("Passwords must match and be at least 8 characters.", "danger")
            return render_template("forgot_password.html", step="reset", username=username)
        db.update_password(user["id"], generate_password_hash(new_pw))
        flash("Password reset successfully. You can now log in.", "success")
        return redirect(url_for("login"))

    return redirect(url_for("forgot_password"))


# ---------------------------------------------------------------
# Chatbot API
# ---------------------------------------------------------------
@app.route("/api/chatbot", methods=["POST"])
def chatbot_api():
    data = request.get_json(silent=True) or {}
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"answer": "Could you type a question? I'm happy to help with diabetes or kidney health questions.", "suggestions": SUGGESTED_QUESTIONS[:3]})
    result = answer_question(question)
    return jsonify(result)


@app.route("/")
def index():
    return render_template(
        "index.html",
        diabetes_auc=round(diabetes_metrics["results"][diabetes_metrics["best_model"]]["roc_auc"] * 100),
        ckd_auc=round(ckd_metrics["results"][ckd_metrics["best_model"]]["roc_auc"] * 100),
        total_models=len(diabetes_metrics["results"]) + len(ckd_metrics["results"]),
    )


@app.route("/diabetes", methods=["GET", "POST"])
def diabetes_form():
    get_session_id()
    if request.method == "GET":
        return render_template("diabetes_form.html")

    form_values = {
        "pregnancies": float(request.form["pregnancies"]),
        "glucose": float(request.form["glucose"]),
        "blood_pressure": float(request.form["blood_pressure"]),
        "skin_thickness": float(request.form["skin_thickness"]),
        "insulin": float(request.form["insulin"]),
        "bmi": float(request.form["bmi"]),
        "pedigree": float(request.form["pedigree"]),
        "age": float(request.form["age"]),
    }

    X = build_diabetes_features(form_values)
    X_scaled = diabetes_scaler.transform(X)
    pred = diabetes_model.predict(X_scaled)[0]
    proba = diabetes_model.predict_proba(X_scaled)[0]
    confidence = float(proba[pred]) * 100

    result = "Diabetic" if pred == 1 else "Not Diabetic"

    top_factors, factor_method = xai.compute_factors(
        diabetes_explainer_kind, diabetes_explainer, diabetes_metrics["feature_names"],
        X_scaled[0], DIABETES_DISPLAY_NAMES, feature_importance=diabetes_metrics.get("feature_importance"),
    )

    sid = get_session_id()
    db.save_diabetes_prediction(sid, form_values, result, confidence, user_id=get_current_user_id())

    session["diabetes_result"] = result
    session["diabetes_confidence"] = confidence
    session["diabetes_inputs"] = form_values
    session["is_diabetic"] = bool(pred == 1)
    session["diabetes_top_factors"] = top_factors
    session["diabetes_factor_method"] = factor_method

    return redirect(url_for("diabetes_result"))


@app.route("/diabetes/result")
def diabetes_result():
    if "diabetes_result" not in session:
        return redirect(url_for("diabetes_form"))
    is_diabetic = session["is_diabetic"]
    return render_template(
        "diabetes_result.html",
        result=session["diabetes_result"],
        confidence=round(session["diabetes_confidence"], 1),
        inputs=session["diabetes_inputs"],
        is_diabetic=is_diabetic,
        precautions=PRECAUTIONS["diabetic" if is_diabetic else "not_diabetic"],
        diabetes_best_model=diabetes_metrics["best_model"],
        top_factors=session.get("diabetes_top_factors", []),
        factor_method=session.get("diabetes_factor_method", "heuristic"),
        urgent_signs=URGENT_SIGNS["diabetic"] if is_diabetic else None,
        diet=DIET_GUIDANCE["diabetic"] if is_diabetic else None,
        specialist=SPECIALIST_SEARCH["diabetic"] if is_diabetic else None,
    )


@app.route("/ckd", methods=["GET", "POST"])
def ckd_form():
    """CKD screening is independently accessible - not gated behind a diabetes result.
    If the user already has a diabetes result this session, relevant fields are pre-filled."""
    get_session_id()
    came_from_diabetes = "diabetes_inputs" in session

    if request.method == "GET":
        if came_from_diabetes:
            prefill = {
                "bp": session["diabetes_inputs"]["blood_pressure"],
                "bgr": session["diabetes_inputs"]["glucose"],
                "age": session["diabetes_inputs"]["age"],
                "dm": 1 if session.get("is_diabetic") else 0,
            }
        else:
            prefill = {"bp": "", "bgr": "", "age": "", "dm": 0}
        return render_template("ckd_form.html", prefill=prefill, came_from_diabetes=came_from_diabetes)

    form_values = {
        "age": float(request.form["age"]),
        "bp": float(request.form["bp"]),
        "sg": float(request.form["sg"]),
        "al": float(request.form["al"]),
        "su": float(request.form["su"]),
        "bgr": float(request.form["bgr"]),
        "bu": float(request.form["bu"]),
        "sc": float(request.form["sc"]),
        "sod": float(request.form["sod"]),
        "pot": float(request.form["pot"]),
        "hemo": float(request.form["hemo"]),
        "htn": int(request.form["htn"]),
        "dm": int(request.form["dm"]),
    }

    X = pd.DataFrame([[form_values[c] for c in CKD_COLS]], columns=CKD_COLS)
    X_scaled = ckd_scaler.transform(X)
    pred = ckd_model.predict(X_scaled)[0]
    proba = ckd_model.predict_proba(X_scaled)[0]
    confidence = float(proba[pred]) * 100

    result = "High Risk" if pred == 1 else "Low Risk"

    top_factors, factor_method = xai.compute_factors(
        ckd_explainer_kind, ckd_explainer, ckd_metrics["feature_names"],
        X_scaled[0], CKD_DISPLAY_NAMES, feature_importance=ckd_metrics.get("feature_importance"),
    )

    sid = get_session_id()
    db.save_ckd_prediction(sid, form_values, result, confidence, user_id=get_current_user_id())

    session["ckd_result"] = result
    session["ckd_confidence"] = confidence
    session["ckd_inputs"] = form_values
    session["ckd_top_factors"] = top_factors
    session["ckd_factor_method"] = factor_method

    return redirect(url_for("ckd_result"))


@app.route("/ckd/result")
def ckd_result():
    if "ckd_result" not in session:
        return redirect(url_for("ckd_form"))
    high_risk = session["ckd_result"] == "High Risk"
    return render_template(
        "ckd_result.html",
        result=session["ckd_result"],
        confidence=round(session["ckd_confidence"], 1),
        inputs=session["ckd_inputs"],
        precautions=PRECAUTIONS["ckd_high_risk" if high_risk else "ckd_low_risk"],
        has_diabetes_result="diabetes_result" in session,
        ckd_best_model=ckd_metrics["best_model"],
        top_factors=session.get("ckd_top_factors", []),
        factor_method=session.get("ckd_factor_method", "heuristic"),
        urgent_signs=URGENT_SIGNS["ckd_high_risk"] if high_risk else None,
        diet=DIET_GUIDANCE["ckd_high_risk"] if high_risk else None,
        specialist=SPECIALIST_SEARCH["ckd_high_risk"] if high_risk else None,
    )


@app.route("/history")
def history():
    sid = get_session_id()
    uid = get_current_user_id()
    hist = db.get_session_history(sid, user_id=uid)
    return render_template("history.html", diabetes_history=hist["diabetes"], ckd_history=hist["ckd"],
                            is_account_history=bool(uid))


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        diabetes_metrics=diabetes_metrics,
        ckd_metrics=ckd_metrics,
        has_diabetes_result="diabetes_result" in session,
        has_ckd_result="ckd_result" in session,
        diabetes_result=session.get("diabetes_result"),
        diabetes_confidence=round(session.get("diabetes_confidence", 0), 1),
        ckd_result=session.get("ckd_result"),
        ckd_confidence=round(session.get("ckd_confidence", 0), 1),
    )


# ---------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------
def _metric_table(model_results, best_model, header_color):
    data = [["Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
    for name, m in model_results.items():
        data.append([
            name, f"{m['accuracy']*100:.1f}%", f"{m['precision']*100:.1f}%",
            f"{m['recall']*100:.1f}%", f"{m['f1']*100:.1f}%", f"{m['roc_auc']*100:.1f}%",
        ])
    t = Table(data, colWidths=[3.6 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]
    for i, name in enumerate(model_results.keys()):
        if name == best_model:
            style.append(("FONTNAME", (0, i + 1), (-1, i + 1), "Helvetica-Bold"))
            style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#EAF2FA")))
    t.setStyle(TableStyle(style))
    return t


@app.route("/report/csv")
def download_csv():
    if "diabetes_result" not in session:
        return redirect(url_for("diabetes_form"))

    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Report generated", datetime.now().strftime("%Y-%m-%d %H:%M")])
    writer.writerow([])
    writer.writerow(["Stage 1: Diabetes Prediction"])
    writer.writerow(["Field", "Value"])
    d_inputs = session["diabetes_inputs"]
    for k, v in d_inputs.items():
        writer.writerow([k, v])
    writer.writerow(["Result", session["diabetes_result"]])
    writer.writerow(["Confidence (%)", round(session["diabetes_confidence"], 1)])

    if "ckd_result" in session:
        writer.writerow([])
        writer.writerow(["Stage 2: Kidney Disease (CKD) Risk Prediction"])
        writer.writerow(["Field", "Value"])
        c_inputs = session["ckd_inputs"]
        for k, v in c_inputs.items():
            writer.writerow([k, v])
        writer.writerow(["Result", session["ckd_result"]])
        writer.writerow(["Confidence (%)", round(session["ckd_confidence"], 1)])

    mem = io.BytesIO(buf.getvalue().encode("utf-8"))
    return send_file(mem, mimetype="text/csv", as_attachment=True,
                      download_name="vitalguard_prediction_values.csv")


@app.route("/report/pdf")
def download_report():
    if "diabetes_result" not in session:
        return redirect(url_for("diabetes_form"))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                             leftMargin=2 * cm, rightMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#1F3864"), fontSize=22)
    subtitle_style = ParagraphStyle("Subtitle", parent=styles["Normal"], textColor=colors.HexColor("#6B7685"), alignment=TA_CENTER, fontSize=10)
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=colors.HexColor("#2E75B6"), spaceBefore=10)
    heading_red = ParagraphStyle("HeadingRed", parent=styles["Heading2"], textColor=colors.HexColor("#A9433D"), spaceBefore=10)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9.5, leading=14)
    result_style_pos = ParagraphStyle("ResultPos", parent=styles["Heading1"], textColor=colors.HexColor("#8A2F28"), alignment=TA_CENTER)
    result_style_neg = ParagraphStyle("ResultNeg", parent=styles["Heading1"], textColor=colors.HexColor("#1F6E4A"), alignment=TA_CENTER)

    elements = []

    # ---------- Cover ----------
    elements.append(Spacer(1, 1.5 * cm))
    elements.append(Paragraph("VitalGuard AI", ParagraphStyle("brand", parent=styles["Normal"], alignment=TA_CENTER, fontSize=13, textColor=colors.HexColor("#2E75B6"), fontName="Helvetica-Bold")))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph("Diabetes &amp; Chronic Kidney Disease", title_style))
    elements.append(Paragraph("Risk Assessment Report", title_style))
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph("Generated by a Two-Stage Machine Learning Screening System", subtitle_style))
    elements.append(Paragraph(f"Report Date: {datetime.now().strftime('%d %B %Y, %I:%M %p')}", subtitle_style))
    elements.append(Spacer(1, 0.6 * cm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#D9E2EC")))
    elements.append(Spacer(1, 0.8 * cm))

    is_diabetic = session["is_diabetic"]
    elements.append(Paragraph("Stage 1 Result", ParagraphStyle("s1", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, textColor=colors.grey)))
    elements.append(Paragraph(session["diabetes_result"], result_style_pos if is_diabetic else result_style_neg))
    elements.append(Paragraph(f"Model confidence: {round(session['diabetes_confidence'], 1)}%",
                               ParagraphStyle("conf", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10)))

    if "ckd_result" in session:
        elements.append(Spacer(1, 0.5 * cm))
        high_risk = session["ckd_result"] == "High Risk"
        elements.append(Paragraph("Stage 2 Result", ParagraphStyle("s2", parent=styles["Normal"], alignment=TA_CENTER, fontSize=11, textColor=colors.grey)))
        elements.append(Paragraph(f"{session['ckd_result']} of CKD", result_style_pos if high_risk else result_style_neg))
        elements.append(Paragraph(f"Model confidence: {round(session['ckd_confidence'], 1)}%",
                                   ParagraphStyle("conf2", parent=styles["Normal"], alignment=TA_CENTER, fontSize=10)))

    elements.append(PageBreak())

    # ---------- Stage 1 details ----------
    elements.append(Paragraph("Stage 1: Diabetes Prediction \u2014 Submitted Values", heading_style))
    d_inputs = session["diabetes_inputs"]
    d_table_data = [["Field", "Value"]] + [
        ["Pregnancies", d_inputs["pregnancies"]],
        ["Glucose (mg/dL)", d_inputs["glucose"]],
        ["Blood Pressure (mm Hg)", d_inputs["blood_pressure"]],
        ["Skin Thickness (mm)", d_inputs["skin_thickness"]],
        ["Insulin (mu U/ml)", d_inputs["insulin"]],
        ["BMI (kg/m\u00b2)", d_inputs["bmi"]],
        ["Diabetes Pedigree Function", d_inputs["pedigree"]],
        ["Age", d_inputs["age"]],
        ["Result", session["diabetes_result"]],
        ["Confidence", f"{round(session['diabetes_confidence'], 1)}%"],
    ]
    t1 = Table(d_table_data, colWidths=[8 * cm, 6 * cm])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    elements.append(t1)
    elements.append(Spacer(1, 0.5 * cm))

    elements.append(Paragraph("Recommended Precautions", heading_style))
    for p in PRECAUTIONS["diabetic" if is_diabetic else "not_diabetic"]:
        elements.append(Paragraph(f"\u2022 {p}", body_style))

    eval_img_path = os.path.join(IMG_DIR, "diabetes_eval.png")
    if os.path.exists(eval_img_path):
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph("Model Evaluation \u2014 Confusion Matrix &amp; ROC Curve", heading_style))
        elements.append(Image(eval_img_path, width=16 * cm, height=6.4 * cm))

    # ---------- Stage 2 details ----------
    if "ckd_result" in session:
        elements.append(PageBreak())
        elements.append(Paragraph("Stage 2: Kidney Disease (CKD) Risk \u2014 Submitted Values", heading_red))
        c_inputs = session["ckd_inputs"]
        c_table_data = [["Field", "Value"]] + [
            ["Age", c_inputs["age"]],
            ["Blood Pressure (mm Hg)", c_inputs["bp"]],
            ["Specific Gravity", c_inputs["sg"]],
            ["Albumin", c_inputs["al"]],
            ["Sugar", c_inputs["su"]],
            ["Blood Glucose Random (mg/dL)", c_inputs["bgr"]],
            ["Blood Urea (mg/dL)", c_inputs["bu"]],
            ["Serum Creatinine (mg/dL)", c_inputs["sc"]],
            ["Sodium (mEq/L)", c_inputs["sod"]],
            ["Potassium (mEq/L)", c_inputs["pot"]],
            ["Hemoglobin (g/dL)", c_inputs["hemo"]],
            ["Hypertension", "Yes" if c_inputs["htn"] == 1 else "No"],
            ["Diabetes Mellitus", "Yes" if c_inputs["dm"] == 1 else "No"],
            ["Result", session["ckd_result"]],
            ["Confidence", f"{round(session['ckd_confidence'], 1)}%"],
        ]
        t2 = Table(c_table_data, colWidths=[8 * cm, 6 * cm])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#A9433D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 0.5 * cm))

        elements.append(Paragraph("Recommended Precautions", heading_red))
        high_risk = session["ckd_result"] == "High Risk"
        for p in PRECAUTIONS["ckd_high_risk" if high_risk else "ckd_low_risk"]:
            elements.append(Paragraph(f"\u2022 {p}", body_style))

        eval_img_path2 = os.path.join(IMG_DIR, "ckd_eval.png")
        if os.path.exists(eval_img_path2):
            elements.append(Spacer(1, 0.4 * cm))
            elements.append(Paragraph("Model Evaluation \u2014 Confusion Matrix &amp; ROC Curve", heading_red))
            elements.append(Image(eval_img_path2, width=16 * cm, height=6.4 * cm))

    # ---------- Appendix: model performance ----------
    elements.append(PageBreak())
    elements.append(Paragraph("Appendix: Full Model Performance Comparison", heading_style))
    elements.append(Spacer(1, 0.2 * cm))
    elements.append(Paragraph("Stage 1 \u2014 Diabetes Prediction Models", ParagraphStyle("s1h", parent=styles["Heading3"], fontSize=11)))
    elements.append(_metric_table(diabetes_metrics["results"], diabetes_metrics["best_model"], colors.HexColor("#1F3864")))
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph("Stage 2 \u2014 CKD Risk Prediction Models", ParagraphStyle("s2h", parent=styles["Heading3"], fontSize=11)))
    elements.append(_metric_table(ckd_metrics["results"], ckd_metrics["best_model"], colors.HexColor("#A9433D")))

    elements.append(Spacer(1, 1 * cm))
    elements.append(HRFlowable(width="100%", color=colors.HexColor("#D9E2EC")))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an academic machine learning mini-project tool and is "
        "not a substitute for professional medical diagnosis. Model predictions are based on statistical patterns "
        "learned from historical datasets and may not account for individual clinical context. Please consult a "
        "qualified doctor for any health concerns or before making any medical decisions.",
        ParagraphStyle("disclaimer", parent=styles["Italic"], fontSize=8, textColor=colors.grey)))

    doc.build(elements)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                      download_name="diabetes_ckd_risk_report.pdf")


# ---------------------------------------------------------------
# Admin routes
# ---------------------------------------------------------------
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    if request.method == "GET":
        return render_template("admin_login.html")

    username = request.form["username"]
    password = request.form["password"]
    admin_row = db.get_admin_by_username(username)
    if admin_row and check_password_hash(admin_row["password_hash"], password):
        session["is_admin"] = True
        session["admin_username"] = username
        return redirect(url_for("admin_dashboard"))

    flash("Invalid username or password.", "danger")
    return redirect(url_for("admin_login"))


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    session.pop("admin_username", None)
    return redirect(url_for("index"))


@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    stats = db.get_dashboard_stats()
    return render_template(
        "admin_dashboard.html",
        stats=stats,
        diabetes_metrics=diabetes_metrics,
        ckd_metrics=ckd_metrics,
        admin_username=session.get("admin_username"),
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug, host="0.0.0.0", port=port)
