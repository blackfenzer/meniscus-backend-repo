from typing import List
from fastapi import (
    APIRouter,
    FastAPI,
    Depends,
    File,
    Response,
    Request,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from itsdangerous import URLSafeTimedSerializer, BadSignature
from starlette.responses import JSONResponse
from datetime import timedelta, datetime
import secrets
from jose import JWTError, jwt
from app.database.session import get_db
from app.models.user import User
from sqlalchemy.orm import Session
import torch
import bentoml
import uuid
from pydantic import BaseModel
import pandas as pd
from app.models.model import ModelMetadata, CSVData
from app.schemas.schemas import PredictRequest, TrainConfig, ModelResponse

router = APIRouter()


router = APIRouter()


@router.post("/train", response_model=ModelResponse)
async def train_model(config: TrainConfig, db: Session = Depends(get_db)):
    """
    Endpoint to train a PyTorch model using provided training configuration.
    This function delegates training to a service which handles both PyTorch and BentoML integration.
    """
    trained_model = train_model_service(config, db)
    return trained_model


@router.post("/predict")
async def predict(input_data: PredictRequest, db: Session = Depends(get_db)):
    """
    Endpoint to get predictions from a trained model.
    """
    predictions = predict_model_service(input_data, db)
    return predictions


@router.get("/models")
async def get_all_models(db: Session = Depends(get_db)):
    """
    Endpoint to list all trained models and their metadata.
    """
    models = get_all_models_service(db)
    return models


@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Endpoint to upload a CSV file which contains data for training.
    The CSV is processed and triggers training.
    """
    try:
        contents = await file.read()
        result = process_csv_training(contents, db)
        return {"status": "success", "detail": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
