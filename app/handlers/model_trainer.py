import os
import torch
import bentoml
from io import BytesIO
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from your_project.database import get_db
from your_project.models import Model
from your_project.utils import validate_train_request_csv, train_model

router = APIRouter()

class ModelTrainer:
    def __init__(self, db: Session):
        self.db = db
    
    async def validate_and_train(self, file: UploadFile) -> dict:
        try:
            temp_file = f"temp_{file.filename}"
            with open(temp_file, "wb") as f:
                f.write(await file.read())

            is_valid = validate_train_request_csv(temp_file)
            if not is_valid:
                os.remove(temp_file)
                return {"valid": False, "message": "Invalid CSV format"}

            model, checkpoint = train_model(temp_file)  # Train the model
            os.remove(temp_file)

            return self.upload_model(model, checkpoint, file.filename)

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

    def upload_model(self, model, checkpoint, filename: str) -> dict:
        try:
            scripted_model = torch.jit.script(model)
            bento_model = bentoml.torchscript.save_model(
                "regression_model",
                scripted_model,
                custom_objects={
                    "scaler": checkpoint["scaler"],
                    "config": checkpoint["config"],
                },
            )

            db_model = Model(
                name="regression_model",
                model_architecture="RegressionNet",
                model_path=filename,
                bentoml_tag=str(bento_model.tag),
                is_active=True,
                csv_id=1,
            )
            self.db.add(db_model)
            self.db.commit()

            return {"status": "success", "bentoml_tag": str(bento_model.tag)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model upload failed: {str(e)}")
