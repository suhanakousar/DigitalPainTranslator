# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# system deps for common Python packages, adjust if your project needs more
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and source
COPY requirements.txt .
# prefer binary wheels where possible
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose port your backend binds to (8000)
EXPOSE 8000

# adjust command to how you start your server, example for uvicorn:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
