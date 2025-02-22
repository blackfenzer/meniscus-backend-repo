import bentoml
from bentoml.io import JSON
import numpy as np
from pydantic import BaseModel
import torch
from typing import Dict, Any
from app.routes.auth2 import check_user, check_admin

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
        # FastAPI will handle auth, this is just an extra layer
        # if not self.validate_request(payload):
        #     return {"error": "Unauthorized"}
        #if (not check_user()):
        #    return bentoml.Response({"error": "User not authorized"}, status=403)

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
                return bentoml.Response({"error": f"Model loading failed: {str(e)}"}, status=500)

        model, scaler = model_cache[model_tag]

        # Process input
        try:
            features = self.extract_features(input_data)
            transformed = scaler.transform([features])
            tensor_input = torch.tensor(transformed, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(tensor_input).numpy().tolist()

            return bentoml.Response({"prediction": prediction}, status=200)

        except Exception as e:
            return bentoml.Response({"error": f"Prediction failed: {str(e)}"}, status=500)
    
    @bentoml.api
    async def delete_model(self, model_tag: str):
        if (not check_admin()):
            return {"error": f"user not authorized"}
        
        # Remove from cache
        if model_tag in model_cache:
            del model_cache[model_tag]
            
        # Optional: Verify deletion from store
        try:
            bentoml.models.delete(model_tag)
        except Exception as e:
            return bentoml.Response({"status": "error", "message": str(e)}, status=500)
            
        return bentoml.Response({"status": "success"}, status=200)

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
