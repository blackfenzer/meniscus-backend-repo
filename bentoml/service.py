import bentoml
import numpy as np
from pydantic import BaseModel
import torch
from typing import Dict, List
from loguru import logger
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
from jose import JWTError, jwt

load_dotenv()

# Configuration and logger setup (same as before)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO")
LOG_PATH = os.getenv("LOG_PATH", "logs/bentoml.log")
SECRET_KEY = "super-secret-key-change-this"
ALGORITHM = "HS256"
Path("logs").mkdir(exist_ok=True)
logger.remove()
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<magenta>{extra[request_id]}</magenta> - "
    "<level>{message}</level>"
)
if ENVIRONMENT == "development":
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        colorize=True,
        format=log_format,
        backtrace=True,
        diagnose=True,
    )
else:
    logger.add(
        LOG_PATH,
        rotation=os.getenv("LOG_ROTATION", "500 MB"),
        retention=os.getenv("LOG_RETENTION", "30 days"),
        compression=os.getenv("LOG_COMPRESSION", "zip"),
        level=LOG_LEVEL,
        serialize=True,
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


# Define Pydantic models for prediction input
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
                model_tag = payload.model_tag
                input_data = payload.input_data

                # Load model and scaler from cache if available
                if model_tag not in model_cache:
                    try:
                        model = bentoml.torchscript.load_model(model_tag)
                        # Access custom objects
                        bento_model = bentoml.models.get(model_tag)
                        scaler = bento_model.custom_objects["scaler"]
                        logger.info("Model and scaler loaded")
                        model.eval()
                        model_cache[model_tag] = (model, scaler)
                    except Exception as e:
                        return {"error": f"Model loading failed: {str(e)}"}, 500

                model, scaler = model_cache[model_tag]

                try:
                    # Extract numeric features in the correct order
                    features = self.extract_features2(input_data)
                    # Convert features to a 2D numpy array (only numeric values, no feature names)
                    transformed = scaler.transform([features])
                    tensor_input = torch.tensor(transformed, dtype=torch.float32)

                    with torch.no_grad():
                        prediction = model(tensor_input).squeeze()
                    # Optionally compute feature importance (if needed)
                    feature_importance = self.get_feature_importance(model, scaler)
                    prediction = prediction.cpu().numpy().tolist()  
                    logger.info("Prediction completed", prediction=prediction)
                    return {
                        "prediction": prediction,
                        "feature_importance": feature_importance,
                    }, 200

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

    def extract_features2(self, input_data: PredictData) -> List[float]:
        """
        Extracts numeric features in the exact order expected by the model.
        The expected order should match the order used when training (and fitting the scaler).
        """
        expected_order = [
            "sex",
            "age",
            "side",
            "BW",
            "Ht",
            "BMI",
            "IKDC_pre",
            "Lysholm_pre",
            "Pre_KL_grade",
            "MM_extrusion_pre",
            "MM_gap",
            "Degenerative_meniscus",
            "medial_femoral_condyle",
            "medial_tibial_condyle",
            "lateral_femoral_condyle",
            "lateral_tibial_condyle",
        ]
        data_dict = input_data.dict()
        try:
            features = [data_dict[feature] for feature in expected_order]
        except KeyError as e:
            logger.error("Missing feature in input data", missing_feature=str(e))
            raise e
        return features

    def get_feature_importance(self, model, scaler, feature_names=None):
        try:
            # Assume the first layer is an nn.Linear layer and extract its weights
            first_param = next(model.parameters())
            weights = first_param.detach().cpu().numpy()
            importance = np.abs(weights).mean(axis=0)
            # If the scaler has feature names, use them; otherwise, use the expected order
            if hasattr(scaler, "feature_names_in_"):
                feature_names = list(scaler.feature_names_in_)
            else:
                feature_names = [
                    "sex",
                    "age",
                    "side",
                    "BW",
                    "Ht",
                    "BMI",
                    "IKDC_pre",
                    "Lysholm_pre",
                    "Pre_KL_grade",
                    "MM_extrusion_pre",
                    "MM_gap",
                    "Degenerative_meniscus",
                    "medial_femoral_condyle",
                    "medial_tibial_condyle",
                    "lateral_femoral_condyle",
                    "lateral_tibial_condyle",
                ]
            sorted_idx = np.argsort(importance)[::-1]
            sorted_features = [feature_names[i] for i in sorted_idx]
            sorted_importance = importance[sorted_idx]
            return {
                feature: float(imp)
                for feature, imp in zip(sorted_features, sorted_importance)
            }
        except Exception as e:
            logger.error("Failed to compute feature importance", error=str(e))
            return {"error": "Feature importance computation failed"}
