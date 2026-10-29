FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Train the (synthetic-data) response model at build time so hybrid/ML ranking works out of the box.
RUN mkdir -p data models docs && python -m scripts.train_model

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
# Set SECRET_KEY, ADMIN_EMAIL and ADMIN_PASSWORD at runtime (see .env.example):
#   docker run -p 8000:8000 --env-file .env trustblood
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
