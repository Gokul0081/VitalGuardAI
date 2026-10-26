"""
chatbot.py
A retrieval-based medical FAQ chatbot for VitalGuard AI.

How it works (honestly, no exaggeration): this is NOT a large language model.
It's a classic NLP information-retrieval technique - every question in a curated
knowledge base is vectorized with TF-IDF, the user's query is vectorized the
same way, and cosine similarity finds the closest matching answer(s). This is
a real, well-established, explainable technique (the same family of methods
used in early search engines and FAQ-matching systems), and it never
hallucinates an answer outside its knowledge base - if nothing matches well
enough, it says so and offers suggestions instead.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------
# Knowledge base: (question variants, answer, category)
# Each entry's "question" is the canonical phrasing; TF-IDF similarity
# handles users phrasing things differently.
# ---------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "q": "What is diabetes?",
        "a": "Diabetes is a chronic condition where the body either doesn't produce enough insulin (Type 1) or can't effectively use the insulin it makes (Type 2), leading to high blood glucose levels over time.",
        "category": "diabetes",
    },
    {
        "q": "What is the difference between Type 1 and Type 2 diabetes?",
        "a": "Type 1 diabetes is an autoimmune condition where the body attacks insulin-producing cells, usually starting in childhood/young adulthood and requiring insulin from diagnosis. Type 2 diabetes develops when the body becomes resistant to insulin or doesn't produce enough, is more common in adults, and is often linked to lifestyle factors - though it can often be managed (and sometimes reversed) with diet, exercise, and medication.",
        "category": "diabetes",
    },
    {
        "q": "What are the symptoms of diabetes?",
        "a": "Common symptoms include excessive thirst, frequent urination, unexplained fatigue, blurred vision, slow-healing wounds, and unexplained weight loss. Many people with Type 2 diabetes have mild or no symptoms early on, which is why screening matters.",
        "category": "diabetes",
    },
    {
        "q": "What foods should a diabetic avoid?",
        "a": "Generally: sugary drinks, refined white bread/rice in large portions, deep-fried snacks, and sweetened desserts. Focus instead on whole grains, lean protein, non-starchy vegetables, and controlled portions. This is general guidance - a dietitian can tailor a plan to you.",
        "category": "diabetes",
    },
    {
        "q": "What is a normal blood glucose level?",
        "a": "A normal fasting blood glucose is typically 70-99 mg/dL. 100-125 mg/dL fasting is considered prediabetic range, and 126 mg/dL or above on two separate tests generally indicates diabetes. Always confirm with a doctor and proper lab testing.",
        "category": "diabetes",
    },
    {
        "q": "Can diabetes be cured?",
        "a": "Type 1 diabetes currently has no cure and requires lifelong insulin management. Type 2 diabetes can sometimes go into remission with significant lifestyle changes (weight loss, diet, exercise), though this varies by individual and should be guided by a doctor - it's rarely accurate to call it fully 'cured.'",
        "category": "diabetes",
    },
    {
        "q": "What is HbA1c?",
        "a": "HbA1c (glycated hemoglobin) is a blood test that shows your average blood glucose level over the past 2-3 months, making it more reliable than a single glucose reading. Below 5.7% is normal, 5.7-6.4% is prediabetic, and 6.5%+ generally indicates diabetes.",
        "category": "diabetes",
    },
    {
        "q": "What is Chronic Kidney Disease CKD?",
        "a": "Chronic Kidney Disease (CKD) is the gradual loss of kidney function over time, where the kidneys become less able to filter waste and excess fluid from the blood. It's often caused or worsened by diabetes and high blood pressure, and is classified into 5 stages based on severity.",
        "category": "ckd",
    },
    {
        "q": "What are the symptoms of kidney disease?",
        "a": "Early CKD often has no symptoms. As it progresses: swelling in legs/ankles/feet, fatigue, difficulty concentrating, poor appetite, changes in urination frequency, muscle cramps, and dry/itchy skin. Advanced stages can cause nausea and shortness of breath - these need urgent medical attention.",
        "category": "ckd",
    },
    {
        "q": "How is diabetes connected to kidney disease?",
        "a": "High blood glucose over time damages the small blood vessels in the kidneys, reducing their ability to filter blood properly. This is called diabetic nephropathy and is one of the leading causes of CKD worldwide - which is exactly why this tool screens for both together.",
        "category": "connection",
    },
    {
        "q": "What foods should someone with kidney disease avoid?",
        "a": "Generally: excess salt, processed/canned foods, and (depending on your stage and your doctor's guidance) high-potassium foods like bananas, oranges, and potatoes. Protein intake may also need to be moderated. This varies significantly by CKD stage, so please follow your nephrologist's specific guidance.",
        "category": "ckd",
    },
    {
        "q": "What is serum creatinine?",
        "a": "Serum creatinine is a waste product from muscle metabolism that healthy kidneys filter out efficiently. A normal range is roughly 0.6-1.3 mg/dL; higher levels can indicate reduced kidney function, since the kidneys aren't clearing it effectively.",
        "category": "ckd",
    },
    {
        "q": "What is eGFR?",
        "a": "eGFR (estimated Glomerular Filtration Rate) estimates how well your kidneys are filtering blood, calculated from creatinine, age, and sex. An eGFR above 90 is normal; below 60 for 3+ months suggests Chronic Kidney Disease, and below 15 indicates kidney failure.",
        "category": "ckd",
    },
    {
        "q": "Is kidney disease reversible?",
        "a": "Early-stage kidney damage can sometimes be slowed or partially improved by controlling the underlying cause (like diabetes or blood pressure). However, advanced CKD (especially stages 4-5) usually isn't reversible, which is why early detection and prompt management matter so much.",
        "category": "ckd",
    },
    {
        "q": "How accurate is this prediction tool?",
        "a": "Our diabetes model achieves about 82.8% ROC-AUC (Logistic Regression) and our CKD model achieves 100% ROC-AUC on held-out test data (Random Forest) - you can see full, real metrics including precision, recall, and confusion matrices on the Dashboard page. That said, this is a screening tool based on statistical patterns, not a diagnostic device - always confirm results with a qualified doctor.",
        "category": "tool",
    },
    {
        "q": "How does the prediction model work?",
        "a": "We train and compare 5 machine learning algorithms (Logistic Regression, KNN, SVM, Random Forest, XGBoost) per stage, tuned with cross-validation, and pick the best by ROC-AUC. The 'What Influenced This Result' section on your result page uses SHAP values - a published explainability technique - to show which of your specific inputs mattered most.",
        "category": "tool",
    },
    {
        "q": "Is my data safe and private?",
        "a": "Yes - no login is required to use the screening tools, and your results are tied only to an anonymous session unless you choose to create an account. If you do create an account, your data is stored securely with hashed passwords and is only used to show you your own history.",
        "category": "tool",
    },
    {
        "q": "What should I do if I get a high-risk result?",
        "a": "Don't panic - this is a screening tool, not a diagnosis. Use the 'Find a Specialist Near You' link on your result page to locate an endocrinologist (for diabetes) or nephrologist (for kidney risk), and bring your PDF report with you - it has all your submitted values for the doctor to review.",
        "category": "tool",
    },
    {
        "q": "What exercise is good for diabetes prevention?",
        "a": "Aim for at least 150 minutes per week of moderate activity (brisk walking, cycling, swimming) - roughly 30 minutes, 5 days a week. This improves insulin sensitivity. Always check with a doctor before starting a new exercise routine, especially if you have existing health conditions.",
        "category": "diabetes",
    },
    {
        "q": "What is hypertension and why does it matter for kidneys?",
        "a": "Hypertension (high blood pressure) puts extra strain on the small blood vessels in the kidneys, damaging their filtering ability over time - much like it damages blood vessels elsewhere in the body. It's one of the top two causes of CKD, alongside diabetes.",
        "category": "ckd",
    },
]

QUESTIONS = [item["q"] for item in KNOWLEDGE_BASE]

_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
_question_vectors = _vectorizer.fit_transform(QUESTIONS)

SUGGESTED_QUESTIONS = [
    "What is the difference between Type 1 and Type 2 diabetes?",
    "How is diabetes connected to kidney disease?",
    "What foods should a diabetic avoid?",
    "How accurate is this prediction tool?",
    "What should I do if I get a high-risk result?",
]

SIMILARITY_THRESHOLD = 0.18
EMERGENCY_KEYWORDS = ["chest pain", "can't breathe", "cannot breathe", "unconscious", "seizure", "suicide", "emergency"]


def answer_question(user_message):
    lowered = user_message.lower()

    if any(kw in lowered for kw in EMERGENCY_KEYWORDS):
        return {
            "answer": "This sounds like it could be a medical emergency. Please contact emergency services or go to the nearest emergency room right away - I'm not able to help with urgent situations like this.",
            "suggestions": [],
            "is_emergency": True,
        }

    user_vec = _vectorizer.transform([user_message])
    sims = cosine_similarity(user_vec, _question_vectors)[0]
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])

    if best_score >= SIMILARITY_THRESHOLD:
        match = KNOWLEDGE_BASE[best_idx]
        # pull 2 more related suggestions from the same category
        related = [item["q"] for i, item in enumerate(KNOWLEDGE_BASE)
                   if item["category"] == match["category"] and i != best_idx][:2]
        return {
            "answer": match["a"],
            "suggestions": related,
            "matched_question": match["q"],
            "confidence": round(best_score * 100, 1),
        }

    return {
        "answer": "I'm not confident I have a good answer to that specific question. I can help with questions about diabetes, kidney disease (CKD), how they're connected, and how this tool works. Try one of these, or rephrase your question:",
        "suggestions": SUGGESTED_QUESTIONS[:4],
        "confidence": round(best_score * 100, 1),
    }
