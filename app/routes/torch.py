import logging
import os
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from app.database.session import get_db
from app.routes.auth2 import get_token
from sqlalchemy.orm import Session
from fastapi import APIRouter, Request, UploadFile, File, Depends, HTTPException
from io import BytesIO
import torch
import bentoml
from app.models.schema import Model, User
from app.core.regression_net import RegressionNet
import requests
from app.schemas.schemas import PredictRequest
from app.handlers.validate_handler import (
    convert_csv_row_ten_types,
    convert_csv_row_types,
)
from jose import JWTError, jwt
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import httpx

from app.routes.auth2 import get_current_user, get_current_role

router = APIRouter()
HOST = os.getenv("BENTOML_HOST")
BENTOML_URL = os.getenv(f"{HOST}/predict", "http://localhost:5000/predict")


class PredictionRequest(BaseModel):
    model_tag: str
    input_data: PredictRequest


@router.post("/upload")
async def upload_model(
    name: str,
    version: str,
    description: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    # current_user: User = Depends(get_current_user)
):
    # Authentication check here (add your logic)
    # if (current_user.role != "user"):
    #     raise HTTPException(status_code=403, detail="User unauthorized")

    # Load and verify model
    try:
        content = await file.read()
        checkpoint = torch.load(
            BytesIO(content), map_location="cpu", weights_only=False
        )

        # Initialize model with your architecture
        model = RegressionNet(input_dim=10, hidden_dim=151, num_layers=2, dropout=0.15)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        scripted_model = torch.jit.script(model)
        # Save to BentoML model store
        bento_model = bentoml.torchscript.save_model(
            name,
            scripted_model,
            custom_objects={
                "scaler": checkpoint["scaler"],
                "config": {
                    "input_dim": 10,
                    "hidden_dim": 151,
                    "num_layers": 2,
                    "dropout": 0.15,
                },
            },
            labels={"version": version, "description": description},
        )

        # Store in database
        db_model = Model(
            name=name,
            model_architecture="RegressionNet",
            model_path=file.filename,
            # model_data=content,
            bentoml_tag=str(bento_model.tag),
            is_active=True,
        )
        db.add(db_model)
        db.commit()

        return {"status": "success", "bentoml_tag": str(bento_model.tag)}

    except Exception as e:
        raise HTTPException(500, f"Model upload failed: {str(e)}")


@router.post("/{model_name}")
async def predict(
    model_name: str,
    prediction_request: PredictionRequest,
    db: Session = Depends(get_db),
    role: User = Depends(get_current_user),
):
    # Extract data from the request
    model_tag = prediction_request.model_tag
    input_data = prediction_request.input_data.dict()
    # input_data = convert_csv_row_ten_types(input_data.values)

    # Retrieve model metadata from the database
    model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found or inactive")

    # Forward the request to the BentoML service
    try:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(f"Received input data: {input_data}")

        async with httpx.AsyncClient() as client:
            response = await client.post(
                BENTOML_URL,
                json={
                    "payload": {
                        "model_tag": model.bentoml_tag,
                        "input_data": input_data,
                    },
                    "headers": {"Authorization": f"Bearer {get_token}"},
                    "YOUR_SECURE_TOKEN": "string",
                },
                timeout=30,
            )
        response.raise_for_status()
        return response.json()

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction service error: {str(e)}"
        )


COOKIE_NAME = "session_token"


@router.post("/{model_name}")
async def predict(
    model_name: str,
    prediction_request: PredictionRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),  # Use your existing dependency
):
    # Verify active user status (already handled in get_current_user)

    # Prepare headers for BentoML
    headers = {
        "Cookie": f"{COOKIE_NAME}={request.cookies.get(COOKIE_NAME)}",
        "X-CSRF-Token": request.cookies.get("csrf_token", ""),
    }
    model_tag = prediction_request.model_tag
    input_data = prediction_request.input_data
    # input_data = convert_csv_row_ten_types(input_data.values)

    # Retrieve model metadata from the database
    model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                BENTOML_URL,
                json={
                    "payload": {
                        "model_tag": model.bentoml_tag,
                        "input_data": {
                            "sex": input_data["sex"],
                            "age": input_data["age"],
                            "side": input_data["side"],
                            "BW": input_data["BW"],
                            "Ht": input_data["Ht"],
                            "BMI": input_data["BMI"],
                            "IKDC pre": input_data["IKDC_pre"],
                            "Lysholm pre": input_data["Lysholm_pre"],
                            "Pre KL grade": input_data["Pre_KL_grade"],
                            "MM extrusion pre": input_data["MM_extrusion_pre"],
                        },
                    },
                    "YOUR_SECURE_TOKEN": "string",
                },
                timeout=30,
            )
        response.raise_for_status()
        return response.json()

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail="Prediction service unavailable")
