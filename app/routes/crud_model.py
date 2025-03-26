import os
from fastapi import APIRouter, HTTPException, Depends
import httpx
from sqlalchemy.orm import Session
from app.schemas.schemas import AllModelResponse, AllModelUpdate, UserSchema
from app.database.session import get_db
from app.models.schema import Model
from jose import jwt
from app.routes.auth2 import get_current_user, protected_route
from loguru import logger

SECRET_KEY = os.getenv("SECRET_KEY") # Store this securely, ideally in environment variables
ALGORITHM = os.getenv("ALGORITHM")
router = APIRouter()
HOST = os.getenv("BENTOML_HOST")
BENTOML_URL = f"{HOST}delete_model"


@router.get("/", response_model=list[AllModelResponse])
def read_models(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(get_current_user),
):
    return (
        db.query(Model).filter(Model.is_active == True).offset(skip).limit(limit).all()
    )


@router.get("/{model_name}", response_model=AllModelResponse)
def read_model(
    model_name: str,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(get_current_user),
):
    model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.put("/models/{model_name}", response_model=AllModelResponse)
def update_model(
    model_name: str,
    model: AllModelUpdate,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(protected_route),
):
    db_model = db.query(Model).filter(Model.name == model_name).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    for key, value in model.dict(exclude_unset=True).items():
        if value is not None and value != "":
            setattr(db_model, key, value)

    try:
        db.commit()
        db.refresh(db_model)
    except Exception as e:
        db.rollback()
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    return db_model


@router.delete("/{model_name}")
async def delete_model(
    model_name: str,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(protected_route),
):
    # Fetch active model from the database
    db_model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Create token data and encode a secure token
    token_data = {
        "user_id": str(user.id),
        "role": str(user.role),
    }
    secure_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

    # Call the Bentoml service for deletion
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BENTOML_URL}",
                json={
                    "payload": {
                        "model_tag": db_model.bentoml_tag,
                        "secure_token": secure_token,
                    },
                },
                timeout=30,
            )
        response.raise_for_status()
    except httpx.RequestError as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=502, detail=f"Prediction service unavailable: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"JSON serialization error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # Delete the model from the database
    try:
        db.delete(db_model)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")

    return {"message": "Model deleted successfully"}
