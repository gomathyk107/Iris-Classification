# Base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Copy app files
COPY ./API /app/api
COPY ./models /app/models
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Set environment variable to point to local MLflow tracking URI
ENV MLFLOW_TRACKING_URI=file:/app/models/mlruns

# Run FastAPI app
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
