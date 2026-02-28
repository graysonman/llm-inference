FROM python:3.11-slim

WORKDIR /app

# System deps (lightweight)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 \
 && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY .env.example ./.env.example

ENV PORT=8000
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
