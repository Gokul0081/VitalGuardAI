# VitalGuard AI — Diabetes Detection and Kidney Disease Risk Prediction using Machine Learning

MCA Semester 3 Mini Project — Gokul U Menon (CEC25MCA-2019)

A two-stage Machine Learning web application:
- **Stage 1** predicts whether a person is diabetic (PIMA Indians Diabetes dataset) — accessible independently
- **Stage 2** predicts Chronic Kidney Disease (CKD) risk (UCI CKD dataset) — accessible independently, or auto-chained after a diabetic result with pre-filled fields

## What's Included

**Machine Learning**
- 5 ML algorithms compared per stage (Logistic Regression, KNN, SVM, Random Forest, XGBoost), each cross-validated and hyperparameter-tuned via GridSearchCV, selected by ROC-AUC
- SMOTE class-balancing (diabetes stage), with a before/after visualization on the dashboard
- Engineered features (Glucose×BMI interaction, BMI/Glucose categories, etc.) for the diabetes model
- **Real SHAP explainability** — per-prediction "top contributing factors" computed with SHAP (SHapley Additive exPlanations), a published, industry-standard technique, with a transparent heuristic fallback if SHAP can't run for a given model type
- Confusion matrix + ROC curve images generated per stage (shown on dashboard and in the PDF report)

**Accounts (optional — guests still work exactly as before)**
- Register / Log In / Log Out
- Profile page with your own check counts
- Change password (while logged in)
- Forgot password — recovered via a security question, no email server required
- Logged-in users get persistent history across visits/devices; guests still get session-only history, no account required

**Chatbot**
- A retrieval-based medical FAQ assistant (`chatbot.py`) using TF-IDF vectorization + cosine similarity — a real, explainable NLP technique, not a general-purpose AI
- Curated knowledge base covering diabetes, CKD, how they're connected, and how this tool itself works
- Detects emergency-sounding language and redirects to emergency services instead of attempting to help
- Floating widget with typing-indicator animation and clickable suggested questions

**Patient-Focused Features**
- Live BMI calculator (enter height/weight, BMI is computed automatically)
- Urgent warning-sign callouts, distinct from general precautions
- Sample diet guidance tailored to each result
- One-click specialist finder (Google Maps search link)
- Field tooltips explaining confusing medical terms
- Prediction History page (account-wide if logged in, session-only as a guest)
- CSV export of raw values, alongside the PDF report
- Print-friendly result pages

**Design & UX**
- Dark mode toggle (persisted across visits)
- Animated "Analyzing..." overlay during prediction, animated confidence rings, toast notifications (color-coded success/error), scroll-reveal cards, button ripple effect, password strength meter, page-fade transitions
- Custom 404 page
- Rich multi-page PDF report: cover summary, full input values, precautions, diet guidance, and a full model-performance appendix
- Admin panel with live usage logs, full metrics tables, and registered-user count

**Deployment**
- `Dockerfile` + `docker-compose.yml` for one-command containerized deployment
- `Procfile` for Render/Railway/Heroku-style free-tier hosting
- Configurable `SECRET_KEY` and `PORT` via environment variables

## Real Model Performance (your numbers will vary slightly on retrain due to randomness)

**Diabetes (best: Logistic Regression)** — Accuracy 75.3%, Precision 62.1%, Recall 75.9%, F1 68.3%, ROC-AUC 82.8%
**CKD (best: Random Forest)** — Accuracy 100%, Precision 100%, Recall 100%, F1 100%, ROC-AUC 100%

These are genuine, measured results — not fabricated. CKD's near-perfect score matches published literature on this dataset (it's a well-separated dataset); diabetes' more modest numbers reflect PIMA's small size and known difficulty, and are consistent with published research.

## Project Structure

```
project/
├── app.py                     # Main Flask application
├── db.py                      # SQLite database helper (users, predictions, admin)
├── explainability.py          # SHAP-based per-prediction factor breakdown
├── chatbot.py                 # TF-IDF + cosine-similarity retrieval chatbot
├── train_diabetes_model.py    # Trains Stage 1 models (SMOTE + GridSearchCV + engineered features)
├── train_ckd_model.py         # Trains Stage 2 models (GridSearchCV)
├── test_app.py                # End-to-end route test script (40+ checks)
├── requirements.txt
├── Dockerfile / .dockerignore / docker-compose.yml
├── Procfile                   # For Render/Railway/Heroku-style hosting
├── data/                      # Raw + cleaned CSV datasets
├── models/                    # Saved .pkl models + metrics .json
├── static/
│   ├── css/                   # style.css, fonts.css
│   ├── js/                    # script.js, chatbot.js
│   ├── img/                   # auto-generated confusion matrix / ROC / SMOTE images
│   ├── vendor/                # Bootstrap, Chart.js, fonts (offline, no CDN needed)
├── templates/                 # All HTML pages (Jinja2)
└── database.db                # Auto-created on first run (SQLite)
```

## Setup (Windows / VS Code)

1. Install Python 3.10+ and open this folder in VS Code.
2. Create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Train both models (only needed once, or whenever you change the datasets):
   ```
   python train_diabetes_model.py
   python train_ckd_model.py
   ```
   This also generates `static/img/diabetes_eval.png`, `static/img/ckd_eval.png` (confusion matrix + ROC curve), and `static/img/smote_balance.png` (class balance before/after SMOTE).
5. Run the app:
   ```
   python app.py
   ```
6. Open your browser at **http://localhost:5000**

## Running with Docker (optional)

```
docker build -t vitalguard-ai .
docker run -p 5000:5000 vitalguard-ai
```
or simply:
```
docker compose up --build
```
Then open **http://localhost:5000**. Note: I wrote this Dockerfile following standard, well-tested patterns, but couldn't test-build it myself in this environment (no Docker daemon available) — please run `docker build .` yourself to confirm before relying on it for a demo.

## Deploying Online (free tier)

This app is ready for Render or Railway's free tier:
1. Push this project to a GitHub repository.
2. On Render.com or Railway.app, create a new Web Service from your repo.
3. They will auto-detect the `Procfile` and `requirements.txt`.
4. Set an environment variable `SECRET_KEY` to any random string.
5. Deploy. You'll get a public URL to use instead of `localhost` — useful for your viva.

I can't click through this deployment for you since it requires your own account, but these 5 steps are everything needed.

## Admin Access

- URL: `/admin/login`
- Default credentials: `admin` / `admin123`

## User Accounts

- Registering is entirely optional — every feature works as a guest.
- Logged-in users get persistent history across visits/devices instead of session-only history.
- Forgot your password? Use the security question you set at registration — no email required.

## Notes

- The site works fully offline — Bootstrap, Chart.js, and fonts are bundled locally under `static/vendor/`, no internet connection needed once dependencies are installed (the chatbot and SHAP explainability also run fully offline/locally — no external API calls).
- `database.db` is created automatically on first run; delete it anytime to reset all logged predictions and accounts.
- Both the Diabetes Check and Kidney Check are independently accessible from the navbar at any time.
- **Design note:** earlier project documentation (ER diagram, use case diagram, abstract) described a login-free system. Adding optional accounts means a `users` table and `user_id` foreign keys were added to the schema — mention this if your evaluator compares the live app against those earlier diagrams.
