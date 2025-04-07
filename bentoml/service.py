import base64
import bentoml
import numpy as np
import pandas as pd
from pydantic import BaseModel
import torch
from typing import Any, Dict, List
from loguru import logger
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv
import shap
from jose import jwt
from jose.exceptions import JWTError, ExpiredSignatureError
import xgboost
from train_handler import train_pipeline_regression, train_xg_boost

load_dotenv()

# Configuration and logger setup (same as before)
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG" if ENVIRONMENT == "development" else "INFO")
LOG_PATH = os.getenv("LOG_PATH", "logs/bentoml.log")
SECRET_KEY = os.getenv(
    "SECRET_KEY"
)  # Store this securely, ideally in environment variables
ALGORITHM = os.getenv("ALGORITHM")
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

feature_names = [
                    "sex",
                    "age",
                    "side",
                    "BW",
                    "Ht",
                    "IKDC_pre",
                    "Lysholm_pre",
                    "Pre_KL_grade",
                    "MM_extrusion_pre",
                    "MM_gap",
                    "Degenerative_meniscus",
                    "medial_femoral_condyle",
                    "medial_tibial_condyle",
                    "lateral_femoral_condyle",
                ]


# Define Pydantic models for prediction input
class PredictData(BaseModel):
    sex: int
    age: int
    side: int
    BW: float
    Ht: float
    IKDC_pre: float
    Lysholm_pre: float
    Pre_KL_grade: float
    MM_extrusion_pre: float
    MM_gap: float
    Degenerative_meniscus: float
    medial_femoral_condyle: float
    medial_tibial_condyle: float
    lateral_femoral_condyle: float

    class Config:
        populate_by_name = True


class PredictInput(BaseModel):
    model_tag: str
    input_data: PredictData
    secure_token: str


class DeleteModelInput(BaseModel):
    model_tag: str
    secure_token: str


class TrainModelInput(BaseModel):
    name: str
    version: str
    description: str
    csv_bytes: str  # base64 encoded CSV content
    model_params: Dict[str, Any]
    optimizer_params: Dict[str, Any]
    batch_size: int
    epochs: int
    secure_token: str


@bentoml.service(
    traffic={"timeout": 30},
    resources={"cpu": "1"},
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
                print("thsi is", payload.secure_token)
                # Verify JWT token
                token_payload = self.verify_token(payload.secure_token)

                if not token_payload:
                    logger.warning("Authentication failed - invalid token")
                    return {"error": "Authentication failed"}, 401

                # if not self.check_authorization(token_payload):
                #     logger.warning(
                #         "Authorization failed",
                #         user_id=token_payload.get("user_id"),
                #         role=token_payload.get("role"),
                #     )
                #     return {"error": "Insufficient permissions"}, 403

                logger.info("Authentication successful", role=token_payload.get("role"))

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
                    # Convert features to a 2D numpy array
                    transformed = scaler.transform([features])
                    tensor_input = torch.tensor(transformed, dtype=torch.float32)

                    with torch.no_grad():
                        prediction = model(tensor_input).squeeze()

                    # Replace feature importance with SHAP values
                    shap_values = self.get_shap_values(model, scaler, features)
                    logger.info("SHAP values shape", shape=np.shape(shap_values), values=shap_values)

                    prediction = prediction.cpu().numpy().tolist()
                    logger.info("Prediction completed", prediction=prediction)
                    logger.info("Feature importance", features = shap_values)
                    return {
                        "prediction": prediction,
                        "feature_importance": shap_values,
                    }, 200

                except Exception as e:
                    logger.error("Prediction failed", error=str(e))
                    return {"error": f"Prediction failed: {str(e)}"}, 500

            except Exception as e:
                logger.critical("Unexpected error in prediction", error=str(e))
                return {"error": "Internal server error"}, 500
            
    @bentoml.api
    async def predictxg(
        self,
        payload: PredictInput,
    ):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                logger.info("Prediction request received")
                model_tag = payload.model_tag
                input_data = payload.input_data

                # Verify JWT token
                token_payload = self.verify_token(payload.secure_token)
                if not token_payload:
                    logger.warning("Authentication failed - invalid token")
                    return {"error": "Authentication failed"}, 401

                logger.info("Authentication successful", role=token_payload.get("role"))

                # Load model and scaler from cache
                if model_tag not in model_cache:
                    try:
                        model = bentoml.xgboost.load_model(model_tag)
                        bento_model = bentoml.models.get(model_tag)
                        scaler = bento_model.custom_objects["scaler"]
                        logger.info("Model and scaler loaded")
                        model_cache[model_tag] = (model, scaler)
                    except Exception as e:
                        return {"error": f"Model loading failed: {str(e)}"}, 500

                model, scaler = model_cache[model_tag]
                
                try:
                    # Extract features as DataFrame and convert to array for scaling
                    features_df = self.extract_features3(input_data)
                    features_array = features_df.to_numpy()  # Convert to numpy array
                    
                    # Scale features
                    transformed = scaler.transform(features_array)  # Now passing an array
                    logger.info("Scaled features", transformed=transformed.tolist())

                    # Make prediction
                    prediction = model.get_booster().predict(xgboost.DMatrix(np.float32(transformed)))
                    logger.info("Raw prediction", prediction=prediction)

                    # Calculate SHAP values using original DataFrame for feature names
                    shap_values = self.get_shap_values_xg(model, scaler, features_df)
                    shap_values = shap_values[0]
                    feature_names = self.get_feature_names()  # Ensure this matches DataFrame columns
                    shap_dict = {feature_names[i]: float(shap_values[i]) for i in range(len(feature_names))}
                    logger.info("SHAP values", values=shap_dict)

                    return {
                        "prediction": prediction.tolist(),
                        "feature_importance": shap_dict,  # Use the correct SHAP dictionary
                    }, 200

                except Exception as e:
                    logger.error("Prediction failed", error=str(e))
                    return {"error": f"Prediction failed: {str(e)}"}, 500

            except Exception as e:
                logger.critical("Unexpected error", error=str(e))
                return {"error": "Internal server error"}, 500

    @bentoml.api
    async def delete_model(self, payload: DeleteModelInput):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                # Verify JWT token
                token_payload = self.verify_token(payload.secure_token)
                if not token_payload:
                    logger.warning("Authentication failed - invalid token")
                    return {"error": "Authentication failed"}, 401

                # Optionally, check for appropriate permissions:
                # if not self.check_authorization(token_payload):
                #     logger.warning("Authorization failed", role=token_payload.get("role"))
                #     return {"error": "Insufficient permissions"}, 403

                logger.info(
                    "Delete model request received", model_tag=payload.model_tag
                )
                if payload.model_tag in model_cache:
                    del model_cache[payload.model_tag]
                    logger.info("Model removed from cache")
                try:
                    bentoml.models.delete(payload.model_tag)
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

    @bentoml.api
    async def get_all_models(self):
        request_id = str(uuid.uuid4())
        with logger.contextualize(request_id=request_id):
            try:
                # Verify JWT token
                # token_payload = self.verify_token(secure_token)
                # if not token_payload:
                #     logger.warning("Authentication failed - invalid token")
                #     return {"error": "Authentication failed"}, 401

                logger.info("Get all models request received")

                try:
                    # Get list of all models from BentoML store
                    models_list = bentoml.models.list()

                    # Format the response with relevant model information
                    models_info = []
                    for model in models_list:
                        model_info = {
                            "tag": str(model.tag),
                            "module": model.info.module,
                            "creation_time": model.info.creation_time.isoformat(),
                        }

                        # Check if model is in cache
                        model_info["in_memory_cache"] = str(model.tag) in model_cache

                        models_info.append(model_info)

                    logger.success(f"Retrieved {len(models_info)} models successfully")
                    return {"models": models_info}, 200

                except Exception as e:
                    logger.error("Failed to retrieve models", error=str(e))
                    return {"error": f"Failed to retrieve models: {str(e)}"}, 500

            except Exception as e:
                logger.critical("Unexpected error in get_all_models", error=str(e))
                return {"error": "Internal server error"}, 500

    @bentoml.api
    async def train_model(self, payload: TrainModelInput):
        # Verify token
        token_payload = self.verify_token(payload.secure_token)
        if not token_payload:
            logger.warning("Authentication failed - invalid token")
            return {"error": "Authentication failed"}, 401

        # Decode the CSV bytes
        try:
            csv_bytes = base64.b64decode(payload.csv_bytes)
        except Exception as e:
            logger.error(f"CSV decoding error: {str(e)}")
            return {"error": f"Failed to decode CSV bytes: {str(e)}"}, 400

        try:
            # Call your training pipeline (this function should match your training logic)
            (
                model,
                train_losses,
                val_losses,
                test_metrics,
                predictions,
                targets,
                scaler,
                input_dim,
            ) = train_pipeline_regression(
                csv_bytes,
                payload.model_params,
                payload.optimizer_params,
                payload.batch_size,
                payload.epochs,
            )
            rmse = test_metrics["rmse"]
            r2 = test_metrics["r2"]
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"error": f"Training failed: {str(e)}"}, 500

        try:
            # Convert model to TorchScript and save with BentoML
            scripted_model = torch.jit.script(model)
            bento_model = bentoml.torchscript.save_model(
                payload.name,
                scripted_model,
                custom_objects={
                    "scaler": scaler,
                    "config": {
                        "input_dim": input_dim,  # Adjust as needed
                        "hidden_dim": payload.model_params.get("hidden_dim"),
                        "num_layers": payload.model_params.get("num_layers"),
                        "dropout": payload.model_params.get("dropout"),
                    },
                },
                labels={"version": payload.version, "description": payload.description},
            )
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            return {"error": f"Model saving failed: {str(e)}"}, 500
        logger.success("Model training and saving successful")
        return {
            "status": "success",
            "bentoml_tag": str(bento_model.tag),
            "rmse": rmse,
            "r2": r2,
        }
    
    @bentoml.api
    async def train_model_xg_boost(self, payload: TrainModelInput):
        # Verify token
        token_payload = self.verify_token(payload.secure_token)
        if not token_payload:
            logger.warning("Authentication failed - invalid token")
            return {"error": "Authentication failed"}, 401

        # Decode the CSV bytes
        try:
            csv_bytes = base64.b64decode(payload.csv_bytes)
        except Exception as e:
            logger.error(f"CSV decoding error: {str(e)}")
            return {"error": f"Failed to decode CSV bytes: {str(e)}"}, 400

        try:
            # Call your training pipeline (this function should match your training logic)
            (
                model,
                train_losses,
                val_losses,
                test_metrics,
                predictions,
                targets,
                scaler,
                input_dim,
            ) = train_xg_boost(
                csv_bytes,
                payload.model_params,
                payload.optimizer_params,
                payload.batch_size,
                payload.epochs,
            )
            rmse = test_metrics["rmse"]
            r2 = test_metrics["r2"]
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {"error": f"Training failed: {str(e)}"}, 500

        try:
            # Convert model to TorchScript and save with BentoML
            bento_model = bentoml.xgboost.save_model(
                payload.name,
                model,
                custom_objects={
                    "scaler": scaler,
                },
                labels={"version": payload.version, "description": payload.description},
            )
        except Exception as e:
            logger.error(f"Model saving failed: {str(e)}")
            return {"error": f"Model saving failed: {str(e)}"}, 500
        logger.success("Model training and saving successful")
        return {
            "status": "success",
            "bentoml_tag": str(bento_model.tag),
            "rmse": rmse,
            "r2": r2,
        }

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
            "IKDC_pre",
            "Lysholm_pre",
            "Pre_KL_grade",
            "MM_extrusion_pre",
            "MM_gap",
            "Degenerative_meniscus",
            "medial_femoral_condyle",
            "medial_tibial_condyle",
            "lateral_femoral_condyle",
        ]
        data_dict = input_data.dict()
        try:
            features = [data_dict[feature] for feature in expected_order]
        except KeyError as e:
            logger.error("Missing feature in input data", missing_feature=str(e))
            raise e
        return features
    
    def extract_features3(self, input_data: PredictData) -> pd.DataFrame:
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
            "IKDC_pre",
            "Lysholm_pre",
            "Pre_KL_grade",
            "MM_extrusion_pre",
            "MM_gap",
            "Degenerative_meniscus",
            "medial_femoral_condyle",
            "medial_tibial_condyle",
            "lateral_femoral_condyle",
        ]
        data_dict = input_data.dict()
    
        try:
            features = {feature: [data_dict[feature]] for feature in expected_order}
        except KeyError as e:
            logger.error("Missing feature in input data", missing_feature=str(e))
            raise e

        return pd.DataFrame(features)

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
                    "IKDC_pre",
                    "Lysholm_pre",
                    "Pre_KL_grade",
                    "MM_extrusion_pre",
                    "MM_gap",
                    "Degenerative_meniscus",
                    "medial_femoral_condyle",
                    "medial_tibial_condyle",
                    "lateral_femoral_condyle",
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

    def get_shap_values(self, model, scaler, features) -> Dict[str, float]:
        """
        Calculate SHAP values for the given features using the provided model and scaler.

        Args:
            model: The PyTorch model
            scaler: The scaler used to normalize features
            features: The raw feature values

        Returns:
            Dictionary mapping feature names to their SHAP values
        """
        try:
            # Create a wrapper function for the PyTorch model
            def f(x):
                with torch.no_grad():
                    tensor_x = torch.tensor(x, dtype=torch.float32)
                    output = model(tensor_x).cpu().numpy()
                    return np.atleast_1d(output)

            # Create a background dataset for SHAP
            # This is typically a sample from your training data
            # For simplicity, we'll use a random sample around the current point
            # In practice, you should use a representative sample from your training data
            n_background = 100
            feature_array = np.array(features)
            background_data = np.random.normal(
                loc=feature_array, scale=0.1, size=(n_background, len(feature_array))
            )
            background_data = scaler.transform(background_data)

            # Initialize the SHAP explainer
            explainer = shap.KernelExplainer(f, background_data)

            # Calculate SHAP values for the current instance
            transformed_features = scaler.transform([features])
            shap_values = explainer.shap_values(transformed_features)[0]

            # Map SHAP values to feature names
            # Assuming you have a way to get feature names in the same order as features
            feature_names = self.get_feature_names()  # Implement this method

            return {
                feature_names[i]: float(shap_values[i])
                for i in range(len(feature_names))
            }

        except Exception as e:
            logger.error(f"SHAP calculation failed: {str(e)}")
            # Fall back to a simpler method or return empty dict
            return {}
        
    def get_shap_values_xg(self, model, scaler, features):
        transformed = scaler.transform(features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)
        return shap_values.tolist() if hasattr(shap_values, "tolist") else shap_values

    def get_feature_names(self) -> List[str]:

        # Implement your logic to get feature names here
        # This should match the order of features in extract_features2
        feature_names = [
            "sex",
            "age",
            "side",
            "BW",
            "Ht",
            "IKDC_pre",
            "Lysholm_pre",
            "Pre_KL_grade",
            "MM_extrusion_pre",
            "MM_gap",
            "Degenerative_meniscus",
            "medial_femoral_condyle",
            "medial_tibial_condyle",
            "lateral_femoral_condyle",
        ]
        return feature_names

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return payload if valid"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            logger.info("Token verified")
            return payload
        except ExpiredSignatureError:
            logger.error("Token expired")
            return None
        except JWTError:
            logger.error("Invalid token")
            return None

    def check_authorization(self, token_payload: Dict[str, Any]) -> bool:
        """Check if the user has appropriate permissions"""
        if not token_payload:
            return False

        # Check role-based permissions
        role = token_payload.get("role")
        # Implement your role-based permission logic here
        if role == "admin":
            return True
        elif role == "user":
            # Add any user-specific restrictions here
            return True
        else:
            return False
