import os
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
from app.database.session import get_db
from app.models.schema import CSVFile

router = APIRouter()


@router.post("/validate-predict-json")
async def validate_predict_request_endpoint(
    json_data: PredictRequest,
) -> Dict[str, bool]:
    data_dict = json_data.dict()
    is_valid = validate_predict_request(data_dict)
    print(is_valid)

    return {"valid": is_valid}


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
        # Read CSV content directly from memory
        csv_content = (await file.read()).decode("utf-8")

        #TODO
        # Validate CSV headers
        # if not validate_train_request_csv(csv_content):
        #     raise HTTPException(status_code=400, detail="Invalid CSV format")
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