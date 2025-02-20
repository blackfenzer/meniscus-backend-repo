import os
import torch
import bentoml
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from typing import Dict, List

from sqlalchemy.orm import Session
from app.schemas.schemas import (
    PredictRequest,
    CSVFileResponse,
    CSVFileListResponse,
    CSVFileBase,
)
from app.handlers.validate_handler import (
    validate_train_request_csv,
    validate_predict_request,
)
from app.handlers.clean_handler import clean_train_request_csv
from app.database.session import get_db
from app.models.schema import CSVFile
from app.models.schema import Model
from app.core.regression_net import RegressionNet
from io import BytesIO
from app.handlers.model_trainer import train_model_from_csv 

router = APIRouter()


@router.post("/validate-predict-json")
async def validate_predict_request_endpoint(
    json_data: PredictRequest,
) -> Dict[str, bool]:
    data_dict = json_data.dict()
    is_valid = validate_predict_request(data_dict)
    print(is_valid)

    return {"valid": is_valid}

## TODO song
@router.post("/validate-train-csv")
async def validate_train_request_endpoint(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> Dict[str, bool]:
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        is_valid = validate_train_request_csv(temp_file)

        # if is_valid:

        os.remove(temp_file)

        return {"valid": is_valid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@router.post("/re-upload")
async def validate_and_upload_csv(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    try:
        ## Read CSV content directly from memory
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        # csv_content = (await file.read()).decode("utf-8")

        # Validate CSV headers
        if not validate_train_request_csv(temp_file):
            os.remove(temp_file)
            raise HTTPException(status_code=400, detail="Invalid CSV format")
        
        with open(temp_file, "r", encoding="utf-8") as f:
            csv_content = f.read()
            
        os.remove(temp_file)
        
        # Save data to DB using the CSVFile model's method
        CSVFile.create_from_csv(db, csv_content)

        return {"valid": True}

    except HTTPException:
        raise
    except Exception as e:

        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@router.get("/all-csv", response_model=CSVFileListResponse)
async def list_csv_files(db: Session = Depends(get_db)):
    csv_files = db.query(CSVFile).all()
    return {
        "files": [
            CSVFileBase(
                id=csv_file.id,
                last_modified_time=csv_file.last_modified_time,
                model_architecture=csv_file.model_architecture,
            )
            for csv_file in csv_files
        ]
    }

## TODO get csv file

## TODO train model and upload model
@router.post("/model_train")
async def model_train_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read CSV content as string
        csv_content = (await file.read()).decode("utf-8")

        # Clean CSV
        cleaned_csv = clean_train_request_csv(csv_content)
        if cleaned_csv is None:
            raise HTTPException(status_code=500, detail="Error cleaning CSV")

        # Save CSV to database
        csv_record = CSVFile.create_from_csv(db, cleaned_csv)

        # Reset file pointer before reading again
        await file.seek(0)
        csv_bytes = await file.read()

        # Train model using cleaned CSV (Implement your model training here)
        model, scaler = train_model_from_csv(csv_bytes)

        # Convert trained model to TorchScript
        scripted_model = torch.jit.script(model)

        # Save model to BentoML
        bento_model = bentoml.torchscript.save_model(
            "regression_model",
            scripted_model,
            custom_objects={
                "scaler": scaler,
                "config": {
                    "input_dim": 10,
                    "hidden_dim": 151,
                    "num_layers": 2,
                    "dropout": 0.15,
                },
            },
            labels={"version": "1.0", "description": "Regression model"},
        )

        # Save model record in DB
        db_model = Model(
            name="regression_model",
            model_architecture="RegressionNet",
            model_path=file.filename,
            bentoml_tag=str(bento_model.tag),
            is_active=True,
            csv_id=csv_record.id,  # Store CSV ID
        )
        db.add(db_model)
        db.commit()

        return {"status": "success", "bentoml_tag": str(bento_model.tag)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
