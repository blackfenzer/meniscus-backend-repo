import bentoml
from bentoml.io import JSON
import numpy as np
from pydantic import BaseModel
from requests import Request
import torch
from typing import Dict, Any, List

from loguru import logger
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from jose import JWTError, jwt

from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# get secret key and hash function

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


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        role = payload.get("role")
        if role is None:
            return None
        return role
    except JWTError:
        return None


# class PredictInput(BaseModel):
#     model_tag: str
#     input_data: Dict[str, Any]


class PredictData(BaseModel):
    sex: int
    age: int
    side: int
    BW: float
    Ht: float
    BMI: float
    IKDC_pre: float
    Lysholm_pre: float
    Pre_KL_grade: float
    MM_extrusion_pre: float

    class Config:
        populate_by_name = True


class PredictInput(BaseModel):
    model_tag: str
    input_data: PredictData


@bentoml.service(
    traffic={"timeout": 30},
    resources={"cpu": "2"},
    http={
        "cors": {
            "enabled": True,
            "access_control_allow_origins": ["*"],
            "access_control_allow_methods": ["*"],
            "access_control_allow_credentials": True,
            "access_control_allow_headers": ["*"],
        },
    }
)
class DynamicRegressionService:

    @bentoml.api(input_spec=PredictInput)
    def predict(self, **payload: Any):

        model_tag = payload["model_tag"]
        input_data = payload["input_data"]
        
        request_id = str(uuid.uuid4())
        logger.bind(request_id=request_id).info("Predict request started", model_tag=model_tag)

        # --- Authentication via gRPC metadata ---
        # NOTE: In gRPC, metadata is passed differently than HTTP headers.
        # BentoML’s gRPC handler makes metadata available via the context.
        # Here we assume that you have a helper function to extract it.
        # For example, you could do:
        #   metadata = bentoml.get_request_metadata()
        # For this demo, we use an empty dict.
        metadata = {}  # Replace with actual metadata extraction

        # --- Model Loading ---
        if model_tag not in model_cache:
            try:
                logger.info("Loading new model", model_tag=model_tag)
                model = bentoml.torchscript.load_model(model_tag)
                bento_model = bentoml.models.get(model_tag)
                scaler = bento_model.custom_objects["scaler"]
                model.eval()
                model_cache[model_tag] = (model, scaler)
                logger.info("Model loaded successfully")
            except Exception as e:
                logger.error("Model loading failed", error=str(e))
                return {"error": f"Model loading failed: {str(e)}"}, 500

        # --- Prediction ---
        model, scaler = model_cache[model_tag]
        try:
            features = self.extract_features(input_data)
            transformed = scaler.transform([features])
            tensor_input = torch.tensor(transformed, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(tensor_input).numpy().tolist()

            logger.info("Prediction completed successfully")
            return {"prediction": prediction}
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return {"error": f"Prediction failed: {str(e)}"}, 500

    @bentoml.api
    async def delete_model(self, model_tag: str, request):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                logger.info("Delete model request received", model_tag=model_tag)

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

    @bentoml.api
    async def delete_all_models(self):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                logger.info("Delete All model request received")
                model_cache.clear()
                logger.info("All models removed from cache")
                models = bentoml.models.list()
                for model in models:
                    logger.success(f"Model {model.tag} deleted successfully")
                    bentoml.models.delete(model.tag)
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

    def extract_features2(self, input_data: PredictData) -> List[float]:
        feature_dict = input_data.dict()

        # Rename feature keys to match expected column names
        column_name_mapping = {
            "Pre_KL_grade": "Pre KL grade",
            "MM_extrusion_pre": "MM extrusion pre",
            "IKDC_pre": "IKDC pre",
            "Lysholm_pre": "Lysholm pre",
            "BW": "BW",  # No change needed, but listed for clarity
            "Ht": "Ht",
            "BMI": "BMI",
            "sex": "sex",
            "age": "age",
            "side": "side",
        }

        # Apply mapping
        renamed_features = {column_name_mapping[k]: v for k, v in feature_dict.items()}

        # Convert to list in the correct order
        feature_values = [renamed_features[col] for col in column_name_mapping.values()]

        return feature_values
