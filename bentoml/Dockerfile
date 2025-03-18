FROM python:3.12.3-slim

WORKDIR /app

# Copy any additional files if needed
# COPY additional_files/ .
# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc python3-dev libgomp1 && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

# Command to serve your BentoML service.
# Replace "Service:svc" with your actual service reference.
CMD ["bentoml", "serve", "service:DynamicRegressionService", "--production"]
# CMD ["bentoml", "serve", "service:DynamicRegressionService", "--production", "--host", "0.0.0.0"]

