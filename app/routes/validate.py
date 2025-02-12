import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
from app.schemas.schemas import PredictRequest
from app.handlers.validate_handler import validate_train_request_csv, validate_predict_request


router = APIRouter()


@router.post("/validate-predict-json")
async def validate_predict_request_endpoint(json_data: PredictRequest) -> Dict[str, bool]:
    data_dict = json_data.dict()
    is_valid = validate_predict_request(data_dict)
    print(is_valid);
    
    return {"valid": is_valid}


@router.post("/validate-train-csv")
async def validate_train_request_endpoint(file: UploadFile = File(...)) -> Dict[str, bool]:
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        is_valid = validate_train_request_csv(temp_file)

        os.remove(temp_file)

        return {"valid": is_valid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")
