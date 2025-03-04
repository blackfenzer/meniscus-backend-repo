FROM bentoml/model-server:latest

WORKDIR /app

# Copy your BentoML service files
COPY . .

# Expose the BentoML service port.
EXPOSE 5000

# Command to serve your BentoML service.
# Replace "Service:svc" with your actual service reference.
CMD ["bentoml", "serve", "service:DynamicRegressionService", "--port", "5010"]
