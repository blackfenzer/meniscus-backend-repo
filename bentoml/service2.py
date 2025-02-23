import bentoml
from bentoml.io import JSON
import numpy as np
from pydantic import BaseModel
import torch
from typing import Dict, Any

# from app.routes.auth2 import check_user, check_admin
from loguru import logger
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Configure logging
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO")
LOG_PATH = os.getenv("LOG_PATH", "logs/bentoml.log")

# Create log directory if needed
Path("logs").mkdir(exist_ok=True)

# Remove default logger and configure Loguru
logger.remove()

# Common log format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{extra[request_id]}</magenta> - "
    "<level>{message}</level>"
)

# Development configuration
if ENVIRONMENT == "development":
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        colorize=True,
        format=log_format,
        backtrace=True,
        diagnose=True,
    )
else:  # Production configuration
    logger.add(
        LOG_PATH,
        rotation=os.getenv("LOG_ROTATION", "500 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        level=LOG_LEVEL,
        serialize=True,  # JSON format for production
        enqueue=True,
        backtrace=True,
        diagnose=False,
    )
model_cache = {}


class PredictInput(BaseModel):
    model_tag: str
    input_data: Dict[str, Any]


@bentoml.service(
    traffic={"timeout": 30},
    resources={"cpu": "2"},
)
class DynamicRegressionService:

    @bentoml.api
    async def predict(self, payload: PredictInput) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                logger.info(
                    "Predict request started",
                    model_tag=payload.model_tag,
                    input_keys=list(payload.input_data.keys()),
                )

                # Authentication
                # if not check_user():
                #     logger.warning("Unauthorized access attempt")
                #     return {"error": "User not authorized"}, 403

                # Model loading
                if payload.model_tag not in model_cache:
                    logger.info("Loading new model", model_tag=payload.model_tag)
                    try:
                        model = bentoml.torchscript.load_model(payload.model_tag)
                        bento_model = bentoml.models.get(payload.model_tag)
                        scaler = bento_model.custom_objects["scaler"]
                        model.eval()
                        model_cache[payload.model_tag] = (model, scaler)
                        logger.success("Model loaded successfully")
                    except Exception as e:
                        logger.error("Model loading failed", error=str(e))
                        return {"error": f"Model loading failed: {str(e)}"}, 500

                # Prediction
                model, scaler = model_cache[payload.model_tag]
                try:
                    features = self.extract_features(payload.input_data)
                    transformed = scaler.transform([features])
                    tensor_input = torch.tensor(transformed, dtype=torch.float32)

                    with torch.no_grad():
                        prediction = model(tensor_input).numpy().tolist()

                    logger.info("Prediction completed successfully")
                    return {"prediction": prediction}

                except Exception as e:
                    logger.error("Prediction failed", error=str(e))
                    return {"error": f"Prediction failed: {str(e)}"}, 500

            except Exception as e:
                logger.critical("Unexpected error in prediction", error=str(e))
                return {"error": "Internal server error"}, 500

    @bentoml.api
    async def delete_model(self, model_tag: str):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                logger.info("Delete model request received", model_tag=model_tag)

                # if not check_admin():
                #     logger.warning("Unauthorized delete attempt")
                #     return {"error": "User not authorized"}, 403

                if model_tag in model_cache:
                    del model_cache[model_tag]
                    logger.info("Model removed from cache")

                try:
                    bentoml.models.delete(model_tag)
                    logger.success("Model deleted permanently")
                    return {"status": "success"}
                except Exception as e:
                    logger.error("Model deletion failed", error=str(e))
                    return {"status": "error", "message": str(e)}, 500

            except Exception as e:
                logger.critical("Unexpected error in deletion", error=str(e))
                return {"error": "Internal server error"}, 500

    def validate_request(self, payload: Dict) -> bool:
        # Add any additional security checks here
        return True  # Replace with actual validation logic

    def extract_features(self, input_data: Dict) -> list:
        return [
            input_data["sex"],
            input_data["age"],
            input_data["side"],
            input_data["BW"],
            input_data["Ht"],
            input_data["BMI"],
            input_data["IKDC pre"],
            input_data["Lysholm pre"],
            input_data["Pre KL grade"],
            input_data["MM extrusion pre"],
        ]
