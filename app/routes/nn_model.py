# from fastapi import (
#     APIRouter,
#     FastAPI,
#     Depends,
#     Response,
#     Request,
#     HTTPException,
#     UploadFile,
#     File,
# )
# from typing import List, Optional
# import grpc
# from app.resources.proto import inference_pb2, inference_pb2_grpc
# from app.resources.proto import management_pb2, management_pb2_grpc
# import torch
# from app.schemas.schemas import PredictRequest
# import base64
# import json

# router = APIRouter()
# # Establish gRPC channels
# inference_channel = grpc.insecure_channel("localhost:7070")
# management_channel = grpc.insecure_channel("localhost:7071")

# # Create gRPC stubs
# inference_stub = inference_pb2_grpc.InferenceAPIsServiceStub(inference_channel)
# management_stub = management_pb2_grpc.ManagementAPIsServiceStub(management_channel)


# @router.get("/all")
# async def list_models():
#     """List all available models in TorchServe."""
#     try:
#         request = management_pb2.ListModelsRequest()
#         response = management_stub.ListModels(request)
#         print("niggra error ", response)
#         models = []
#         if response.model_version_details:
#             for model_version in response.model_version_details:
#                 models.append(
#                     {
#                         "model_name": model_version.model_name,
#                         "model_version": model_version.version,
#                         "model_url": model_version.model_url,
#                         "runtime": model_version.runtime,
#                         "batch_size": model_version.batch_size,
#                         "min_workers": model_version.min_worker,
#                         "max_workers": model_version.max_worker,
#                         "loading_workers": model_version.loading_workers,
#                         "workers": model_version.workers,
#                         "status": model_version.status,
#                     }
#                 )
#         else:
#             raise HTTPException(status_code=404, detail="No models found.")

#         return {"models": models}
#     except grpc.RpcError as e:
#         status_code = e.code()
#         detail = e.details() or str(e)
#         raise HTTPException(
#             status_code=status_code.value[0], detail=f"gRPC error: {detail}"
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @router.post("/models/register")
# async def register_model(
#     model_name: str, model_url: str, initial_workers: int = 1, synchronous: bool = True
# ):
#     """
#     Register a new model with TorchServe via gRPC.

#     Args:
#         model_name (str): The name to assign to the model in TorchServe.
#         model_url (str): The URL or path to the .mar file in the model store.
#         initial_workers (int, optional): Number of initial workers to assign. Defaults to 1.
#         synchronous (bool, optional): Whether the registration should be synchronous. Defaults to True.

#     Returns:
#         dict: A success message if the model is registered successfully.
#     """
#     try:
#         # Create the RegisterModelRequest
#         request = management_pb2.RegisterModelRequest(
#             model_name=model_name,
#             url=model_url,
#             initial_workers=initial_workers,
#             synchronous=synchronous,
#         )

#         # Send the request to TorchServe
#         response = management_stub.RegisterModel(request)

#         return {
#             "status": "success",
#             "message": f"Model '{model_name}' registered successfully.",
#         }

#     except grpc.RpcError as e:
#         # Handle gRPC errors
#         status_code = e.code()
#         detail = e.details() or str(e)
#         raise HTTPException(
#             status_code=500, detail=f"gRPC error: {status_code.name} - {detail}"
#         )
#     except Exception as e:
#         # Handle other exceptions
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# @router.delete("/models/{model_name}")
# async def unregister_model(model_name: str):
#     """Unregister a model from TorchServe."""
#     try:
#         request = management_pb2.UnregisterModelRequest(model_name=model_name)
#         response = management_stub.UnregisterModel(request)
#         return {
#             "status": "success",
#             "message": f"Model {model_name} unregistered successfully",
#         }
#     except grpc.RpcError as e:
#         raise HTTPException(
#             status_code=500, detail=f"Failed to unregister model: {str(e)}"
#         )


# @router.post("/predictions/{model_name}")
# async def predict(model_name: str, request: PredictRequest):
#     """Make predictions using a specified model."""
#     try:
#         # Convert input data to the format expected by TorchServe
#         inference_request = inference_pb2.PredictionsRequest(
#             model_name=model_name, input=request.input
#         )

#         response = inference_stub.Predictions(inference_request)

#         # Parse the prediction response
#         try:
#             prediction_result = json.loads(response.prediction)
#         except json.JSONDecodeError:
#             prediction_result = response.prediction

#         return {"model_name": model_name, "prediction": prediction_result}
#     except grpc.RpcError as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# @router.post("/models/{model_name}/train")
# async def train_model(
#     model_name: str,
#     training_data: UploadFile = File(...),
#     epochs: Optional[int] = 10,
#     batch_size: Optional[int] = 32,
# ):
#     """Train or fine-tune a model."""
#     try:
#         # Read training data
#         data = await training_data.read()

#         # Create training request
#         request = management_pb2.TrainingRequest(
#             model_name=model_name,
#             training_data=data,
#             training_params=json.dumps({"epochs": epochs, "batch_size": batch_size}),
#         )

#         # Start training
#         response = management_stub.TrainModel(request)

#         return {
#             "status": "success",
#             "message": f"Training started for model {model_name}",
#             "training_id": response.training_id,
#         }
#     except grpc.RpcError as e:
#         raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


# @router.get("/models/{model_name}/status")
# async def get_model_status(model_name: str):
#     """Get the current status of a model."""
#     try:
#         request = management_pb2.DescribeModelRequest(model_name=model_name)
#         response = management_stub.DescribeModel(request)

#         return {
#             "model_name": model_name,
#             "status": response.status,
#             "workers": response.workers,
#             "version": response.version,
#         }
#     except grpc.RpcError as e:
#         raise HTTPException(
#             status_code=500, detail=f"Failed to get model status: {str(e)}"
#         )
