FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ✅ Correct: copy requirements without "backend/" prefix
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ✅ Correct: copy all backend files
COPY . .

EXPOSE 8000

# ✅ Hot reload for development
CMD ["uvicorn", "chatbot_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "/app"]