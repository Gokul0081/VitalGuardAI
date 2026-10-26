"""
test_app.py - sanity check every route works end-to-end using Flask's test client.
"""
import sys
sys.path.insert(0, ".")
from app import app

def check(name, resp, expect_status=200):
    status_ok = resp.status_code == expect_status
    print(f"[{'OK' if status_ok else 'FAIL'}] {name} -> {resp.status_code} (expected {expect_status})")
    return status_ok

all_ok = True

# ---- Scenario A: standalone CKD check, no diabetes check first ----
with app.test_client() as c:
    all_ok &= check("GET / ", c.get("/"))
    r = c.get("/ckd")
    all_ok &= check("GET /ckd standalone (no diabetes session)", r)
    body = r.get_data(as_text=True)
    print("      -> shows 'Standalone Check' pill?", "Standalone Check" in body)

    ckd_payload = {
        "age": "60", "bp": "90", "sg": "1.010", "al": "3", "su": "2",
        "bgr": "180", "bu": "55", "sc": "2.1", "sod": "138", "pot": "4.8",
        "hemo": "10.2", "htn": "1", "dm": "1",
    }
    r = c.post("/ckd", data=ckd_payload, follow_redirects=False)
    all_ok &= check("POST /ckd standalone (redirect)", r, expect_status=302)
    r = c.get("/ckd/result")
    all_ok &= check("GET /ckd/result standalone", r)

    r = c.get("/report/pdf", follow_redirects=False)
    all_ok &= check("GET /report/pdf without diabetes result -> redirect", r, expect_status=302)

# ---- Scenario B: full chained flow ----
with app.test_client() as c:
    all_ok &= check("GET /diabetes (form)", c.get("/diabetes"))

    diabetic_payload = {
        "pregnancies": "6", "glucose": "170", "blood_pressure": "88",
        "skin_thickness": "40", "insulin": "200", "bmi": "38.5",
        "pedigree": "0.9", "age": "55",
    }
    r = c.post("/diabetes", data=diabetic_payload, follow_redirects=False)
    all_ok &= check("POST /diabetes (redirect)", r, expect_status=302)
    r = c.get("/diabetes/result")
    all_ok &= check("GET /diabetes/result", r)
    body = r.get_data(as_text=True)
    print("      -> contains 'Diabetic'?", "Diabetic" in body)
    print("      -> shows precautions?", "Recommended Precautions" in body)

    r = c.get("/ckd")
    all_ok &= check("GET /ckd (chained, pre-filled)", r)
    body = r.get_data(as_text=True)
    print("      -> shows 'Stage 2 of 2' pill?", "Stage 2 of 2" in body)

    ckd_payload = {
        "age": "60", "bp": "90", "sg": "1.010", "al": "3", "su": "2",
        "bgr": "180", "bu": "55", "sc": "2.1", "sod": "138", "pot": "4.8",
        "hemo": "10.2", "htn": "1", "dm": "1",
    }
    r = c.post("/ckd", data=ckd_payload, follow_redirects=False)
    all_ok &= check("POST /ckd chained (redirect)", r, expect_status=302)
    r = c.get("/ckd/result")
    all_ok &= check("GET /ckd/result chained", r)
    body = r.get_data(as_text=True)
    print("      -> contains 'Risk'?", "Risk" in body)

    r = c.get("/dashboard")
    all_ok &= check("GET /dashboard", r)
    body = r.get_data(as_text=True)
    print("      -> contains 'ROC-AUC'?", "ROC-AUC" in body)

    r = c.get("/report/pdf")
    all_ok &= check("GET /report/pdf (full report)", r)
    print("      -> content-type:", r.content_type, "| size:", len(r.data), "bytes")

    r = c.get("/admin/dashboard", follow_redirects=False)
    all_ok &= check("GET /admin/dashboard (not logged in -> redirect)", r, expect_status=302)

    r = c.get("/admin/login")
    all_ok &= check("GET /admin/login", r)

    r = c.post("/admin/login", data={"username": "admin", "password": "wrongpass"}, follow_redirects=False)
    all_ok &= check("POST /admin/login (wrong pw -> redirect back)", r, expect_status=302)

    r = c.post("/admin/login", data={"username": "admin", "password": "admin123"}, follow_redirects=False)
    all_ok &= check("POST /admin/login (correct -> redirect)", r, expect_status=302)

    r = c.get("/admin/dashboard")
    all_ok &= check("GET /admin/dashboard (logged in)", r)
    body = r.get_data(as_text=True)
    print("      -> contains 'Diabetes Checks Run'?", "Diabetes Checks Run" in body)
    print("      -> contains 'ROC-AUC'?", "ROC-AUC" in body)

    r = c.get("/history")
    all_ok &= check("GET /history (with predictions)", r)
    body = r.get_data(as_text=True)
    print("      -> shows diabetes history entry?", "Diabetic" in body or "Not Diabetic" in body)

    r = c.get("/report/csv")
    all_ok &= check("GET /report/csv", r)
    print("      -> content-type:", r.content_type, "| size:", len(r.data), "bytes")

    r = c.get("/this-page-does-not-exist")
    all_ok &= check("GET unknown route -> 404 page", r, expect_status=404)
    body = r.get_data(as_text=True)
    print("      -> shows custom 404 content?", "wrong turn" in body)

# ---- Scenario C: history page with zero predictions ----
with app.test_client() as c:
    r = c.get("/history")
    all_ok &= check("GET /history (empty state)", r)
    body = r.get_data(as_text=True)
    print("      -> shows empty-state message?", "No predictions yet" in body)

# ---- Scenario D: user registration, login, profile, password change ----
with app.test_client() as c:
    r = c.get("/register")
    all_ok &= check("GET /register", r)

    reg_payload = {
        "username": "testuser1", "email": "testuser1@example.com",
        "password": "SecurePass123", "confirm_password": "SecurePass123",
        "security_question": "What city were you born in?", "security_answer": "Kochi",
    }
    r = c.post("/register", data=reg_payload, follow_redirects=False)
    all_ok &= check("POST /register (success -> redirect)", r, expect_status=302)

    r = c.get("/profile")
    all_ok &= check("GET /profile (logged in after register)", r)
    body = r.get_data(as_text=True)
    print("      -> shows username?", "testuser1" in body)

    r = c.get("/logout", follow_redirects=False)
    all_ok &= check("GET /logout (-> redirect)", r, expect_status=302)

    r = c.get("/profile", follow_redirects=False)
    all_ok &= check("GET /profile after logout (-> redirect to login)", r, expect_status=302)

    r = c.post("/login", data={"identifier": "testuser1", "password": "wrongpass"}, follow_redirects=False)
    all_ok &= check("POST /login (wrong password -> redirect back)", r, expect_status=302)

    r = c.post("/login", data={"identifier": "testuser1", "password": "SecurePass123"}, follow_redirects=False)
    all_ok &= check("POST /login (correct -> redirect)", r, expect_status=302)

    r = c.get("/profile")
    all_ok &= check("GET /profile (logged in again)", r)

    pw_payload = {"current_password": "SecurePass123", "new_password": "NewPass456", "confirm_new_password": "NewPass456"}
    r = c.post("/profile/change-password", data=pw_payload, follow_redirects=False)
    all_ok &= check("POST /profile/change-password (-> redirect)", r, expect_status=302)

    r = c.get("/logout", follow_redirects=False)
    r = c.post("/login", data={"identifier": "testuser1", "password": "NewPass456"}, follow_redirects=False)
    all_ok &= check("POST /login (with new password -> redirect)", r, expect_status=302)

    # linked prediction history
    diabetic_payload = {
        "pregnancies": "6", "glucose": "170", "blood_pressure": "88",
        "skin_thickness": "40", "insulin": "200", "bmi": "38.5",
        "pedigree": "0.9", "age": "55",
    }
    c.post("/diabetes", data=diabetic_payload)
    r = c.get("/history")
    all_ok &= check("GET /history (account-linked)", r)
    body = r.get_data(as_text=True)
    print("      -> account history shows prediction?", "Diabetic" in body)

# ---- Scenario E: forgot password flow ----
with app.test_client() as c:
    c.post("/register", data={
        "username": "resetuser", "email": "resetuser@example.com",
        "password": "OldPass123", "confirm_password": "OldPass123",
        "security_question": "What is your mother's maiden name?", "security_answer": "Smith",
    })
    c.get("/logout")

    r = c.post("/forgot-password", data={"step": "find", "identifier": "resetuser"})
    all_ok &= check("POST /forgot-password (find)", r)
    body = r.get_data(as_text=True)
    print("      -> shows security question?", "maiden name" in body)

    r = c.post("/forgot-password", data={"step": "answer", "username": "resetuser", "security_answer": "smith"})
    all_ok &= check("POST /forgot-password (correct answer)", r)
    body = r.get_data(as_text=True)
    print("      -> shows reset form?", "New Password" in body)

    r = c.post("/forgot-password", data={"step": "answer", "username": "resetuser", "security_answer": "wronganswer"})
    body = r.get_data(as_text=True)
    print("      -> wrong answer rejected?", "New Password" not in body)

    r = c.post("/forgot-password", data={
        "step": "reset", "username": "resetuser",
        "new_password": "BrandNewPass789", "confirm_new_password": "BrandNewPass789",
    }, follow_redirects=False)
    all_ok &= check("POST /forgot-password (reset -> redirect to login)", r, expect_status=302)

    r = c.post("/login", data={"identifier": "resetuser", "password": "BrandNewPass789"}, follow_redirects=False)
    all_ok &= check("POST /login (with reset password -> redirect)", r, expect_status=302)

# ---- Scenario F: chatbot API ----
with app.test_client() as c:
    r = c.post("/api/chatbot", json={"message": "What is the difference between type 1 and type 2 diabetes"})
    all_ok &= check("POST /api/chatbot (known question)", r)
    data = r.get_json()
    print("      -> matched a relevant answer?", "Type 1" in data.get("answer", "") and "Type 2" in data.get("answer", ""))
    print("      -> includes suggestions?", len(data.get("suggestions", [])) > 0)

    r = c.post("/api/chatbot", json={"message": "asdkjaslkdj random gibberish xyz"})
    all_ok &= check("POST /api/chatbot (gibberish -> low confidence fallback)", r)
    data = r.get_json()
    print("      -> gives fallback with suggestions?", len(data.get("suggestions", [])) > 0)

    r = c.post("/api/chatbot", json={"message": "I am having chest pain and can't breathe"})
    all_ok &= check("POST /api/chatbot (emergency keywords)", r)
    data = r.get_json()
    print("      -> flagged as emergency?", data.get("is_emergency") is True)

    r = c.post("/api/chatbot", json={"message": ""})
    all_ok &= check("POST /api/chatbot (empty message)", r)

print()
print("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED")
