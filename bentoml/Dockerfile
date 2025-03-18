FROM bentoml/model-server:latest

WORKDIR /app

# Install additional dependencies
RUN pip install --no-cache-dir numpy pandas torch pydantic loguru python-dotenv shap python-jose

# Copy your BentoML service file
COPY service.py .

# Copy any additional files if needed
# COPY additional_files/ .

# Expose the BentoML service port
EXPOSE 5000

# Command to serve your BentoML service
CMD ["bentoml", "serve", "service:DynamicRegressionService", "--port", "5000"]