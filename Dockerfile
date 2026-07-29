FROM python:3.11-slim

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ .

COPY models/ /app/models/

ENV PORT=8080
ENV ARTIFACT_DIR=/app/models/v1

EXPOSE 8080

CMD ["sh","-c","uvicorn api:app --host 0.0.0.0 --port ${PORT}"]