from pydantic import BaseModel
from typing import List, Optional


class ModelConfig(BaseModel):
    input_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1


class LoadModelRequest(BaseModel):
    model_name: str
    model_path: str
    model_config: Optional[ModelConfig] = None


class PredictionRequest(BaseModel):
    model_name: str
    data: List[List[float]]


class PredictionResponse(BaseModel):
    status: str
    predictions: Optional[List[float]] = None
    message: Optional[str] = None


class ModelResponse(BaseModel):
    status: str
    message: str


class ListModelsResponse(BaseModel):
    models: List[str]


class ModelConfig(BaseModel):
    input_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1


class LoadModelRequest(BaseModel):
    model_name: str
    model_path: str
    model_config: ModelConfig = ModelConfig()


class PredictRequest(BaseModel):
    model_name: str
    data: List[List[float]]


class PredictResponse(BaseModel):
    status: str
    predictions: List[float] = None
    message: str = None
