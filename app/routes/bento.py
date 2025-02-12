import os
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
import httpx
from itsdangerous import URLSafeTimedSerializer, BadSignature
from starlette.responses import JSONResponse
from datetime import timedelta, datetime
import secrets
from jose import JWTError, jwt
from app.database.session import get_db
from sqlalchemy.orm import Session
import torch
import bentoml
import uuid
from pydantic import BaseModel
import pandas as pd
from app.schemas.schemas import PredictRequest, TrainConfig, ModelResponse


router = APIRouter()
async def get_bentoml_client():
    async with httpx.AsyncClient(base_url=os.getenv.BENTOML_HOST) as client:
        yield client

@router.post("/train", response_model=ModelResponse)
async def train_model(config: TrainConfig, db: Session = Depends(get_db)):
    try:
        trained_model = train_model_service(config, db)
        return trained_model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def predict(input_data: PredictRequest, db: Session = Depends(get_db)):
    try:
        predictions = predict_model_service(input_data, db)
        return {"predictions": predictions}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelResponse])
async def get_all_models(db: Session = Depends(get_db)):
    return get_all_models_service(db)


# @router.post("/upload-csv")
# async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
#     """
#     Endpoint to upload a CSV file which contains data for training.
#     The CSV is processed and triggers training.
#     """
#     try:
#         contents = await file.read()
#         result = process_csv_training(contents, db)
#         return {"status": "success", "detail": result}
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=str(e))
