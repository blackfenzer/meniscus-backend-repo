import bentoml
from bentoml.io import JSON
import numpy as np
from pydantic import BaseModel
import torch
from typing import Dict, Any, List

from loguru import logger
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

# get secret key and hash function

# Configure logging
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO")
LOG_PATH = os.getenv("LOG_PATH", "logs/bentoml.log")

SECRET_KEY = "super-secret-key-change-this"
ALGORITHM = "HS256"

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
    MM_gap: float
    Degenerative_meniscus: float
    medial_femoral_condyle: float
    medial_tibial_condyle: float
    lateral_femoral_condyle: float
    lateral_tibial_condyle: float

    class Config:
        populate_by_name = True


class PredictInput(BaseModel):
    model_tag: str
    input_data: PredictData


@bentoml.service(
    traffic={"timeout": 30},
    resources={"cpu": "2"},
)
class DynamicRegressionService:

    @bentoml.api
    async def predict(
        self,
        payload: PredictInput,
    ):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                logger.info("Prediction request received")
        # FastAPI will handle auth, this is just an extra layer
        # if not self.validate_request(payload):
        #     return {"error": "Unauthorized"}

                model_tag = payload.model_tag
                input_data = payload.input_data

                # Load model with caching
                if model_tag not in model_cache:
                    try:
                        model = bentoml.torchscript.load_model(model_tag)

                        # Access custom objects
                        bento_model = bentoml.models.get(model_tag)
                        scaler = bento_model.custom_objects["scaler"]
                        print(scaler)
                        # Set the model to evaluation mode
                        model.eval()

                        # Cache the model and scaler
                        model_cache[model_tag] = (model, scaler)
                    except Exception as e:
                        return {"error": f"Model loading failed: {str(e)}"}, 500

                model, scaler = model_cache[model_tag]

                # Process input
                try:
                    features = self.extract_features2(input_data)
                    transformed = scaler.transform([features])
                    tensor_input = torch.tensor(transformed, dtype=torch.float32)

                    with torch.no_grad():
                        prediction = model(tensor_input).numpy().tolist()
                    logger.info("Prediction completed", prediction=prediction)
                    return {"prediction": prediction}, 200

                except Exception as e:
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
            input_data["MM gap"],
            input_data["Degenerative meniscus"],
            input_data["medial femoral condyle"],
            input_data["medial tibial condyle"],
            input_data["lateral femoral condyle"],
            input_data["lateral tibial condyle"],
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
            "MM_gap": "MM gap",
            "Degenerative_meniscus": "Degenerative meniscus",
            "medial_femoral_condyle": "medial femoral condyle",
            "medial_tibial_condyle": "medial tibial condyle",
            "lateral_femoral_condyle": "lateral femoral condyle",
            "lateral_tibial_condyle": "lateral tibial condyle",
        }

        # Apply mapping
        renamed_features = {column_name_mapping[k]: v for k, v in feature_dict.items()}

        # Convert to list in the correct order
        feature_values = [renamed_features[col] for col in column_name_mapping.values()]

        return feature_values
