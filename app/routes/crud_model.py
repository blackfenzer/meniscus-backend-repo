# Add these to your FastAPI router
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from app.schemas.schemas import AllModelResponse, AllModelUpdate
from app.database.session import get_db
from app.models.schema import Model
import bentoml

router = APIRouter()


# @router.post("/models/", response_model=ModelResponse)
# def create_model(model: ModelCreate, db: Session = Depends(get_db)):
#     db_model = Model(**model.dict(), created_at=datetime.now())
#     db.add(db_model)
#     try:
#         db.commit()
#         db.refresh(db_model)
#     except Exception as e:
#         db.rollback()
#         raise HTTPException(status_code=400, detail=str(e))
#     return db_model


@router.get("/", response_model=list[AllModelResponse])
def read_models(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return (
        db.query(Model).filter(Model.is_active == True).offset(skip).limit(limit).all()
    )


@router.get("/{model_name}", response_model=AllModelResponse)
def read_model(model_name: str, db: Session = Depends(get_db)):
    model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.put("/models/{model_name}", response_model=AllModelResponse)
def update_model(model_name: str, model: AllModelUpdate, db: Session = Depends(get_db)):
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
        raise HTTPException(status_code=400, detail=str(e))
    return db_model


@router.delete("/{model_name}")
def delete_model(model_name: str, db: Session = Depends(get_db)):
    db_model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        # Delete from BentoML model store
        bentoml.models.delete(db_model.bentoml_tag)
    except bentoml.exceptions.NotFound:
        # Model already deleted from BentoML store
        pass
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete model from BentoML store: {str(e)}",
        )

    try:
        # Deactivate in database
        db.delete(db_model)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")

    return {"message": "Model deleted successfully"}
