FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py config.py ./
COPY src ./src

EXPOSE 8000

CMD ["gunicorn", "-b", "0.0.0.0:8000", "api_server:create_app()"]
