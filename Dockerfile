# Build stage
FROM python:3.8-slim-bookworm as builder

WORKDIR /app
COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential \
    && pip install --user -r requirements.txt

# Runtime stage
FROM python:3.8-slim-bookworm

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH

CMD ["python3", "app.py"]
