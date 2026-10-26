# VitalGuard AI - Dockerfile
# Builds a self-contained image with pre-trained models included.
# Build:  docker build -t vitalguard-ai .
# Run:    docker run -p 5000:5000 vitalguard-ai
# Then open http://localhost:5000

FROM python:3.11-slim

WORKDIR /app

# System deps needed by matplotlib/xgboost wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Always retrain inside the container, so the saved .pkl files are guaranteed
# to match the exact scikit-learn/xgboost versions installed here (avoids
# any pickle version-mismatch issues across environments)
RUN python train_diabetes_model.py && python train_ckd_model.py

ENV FLASK_DEBUG=false
EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
