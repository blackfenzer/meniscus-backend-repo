import csv
import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict

router = APIRouter()

INPUT_COLUMNS = ['sex', 'age', 'side', 'BW', 'Ht', 'BMI', 'IKDC pre', 'Lysholm pre', 'Pre KL grade', 'MM extrusion pre']

class JSONData(BaseModel):
    sex: int
    age: int
    side: int
    BW: float
    Ht: float
    BMI: float
    IKDC_pre: float = Field(..., alias="IKDC pre")
    Lysholm_pre: float = Field(..., alias="Lysholm pre")
    Pre_KL_grade: int = Field(..., alias="Pre KL grade")
    MM_extrusion_pre: float = Field(..., alias="MM extrusion pre")


@router.post("/validate-csv")
async def validate_csv_endpoint(file: UploadFile = File(...)) -> Dict[str, bool]:
    try:
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        is_valid = validate_csv(temp_file)

        os.remove(temp_file)

        return {"valid": is_valid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")


@router.post("/validate-json")
async def validate_json_endpoint(json_data: JSONData) -> Dict[str, bool]:
    data_dict = json_data.dict()
    is_valid = validate_variables(data_dict)
    
    return {"valid": is_valid}


def validate_csv(file_path: str) -> bool:
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            if not set(INPUT_COLUMNS).issubset(reader.fieldnames):
                return False

            for row in reader:
                converted_row = convert_csv_row_types(row)
                if not validate_variables(converted_row):
                    return False

    except Exception as e:
        print(f"Exception: {e}")
        return False

    return True


def validate_variables(data: dict) -> bool:
    try:
        if data['sex'] not in [0, 1]:
            return False
        if not (0 <= data['age'] <= 120):
            return False
        if data['side'] not in [1, 2]:
            return False

        numeric_fields = [field for field, type_ in JSONData.__annotations__.items()
                          if type_ in (int, float) and field not in ['sex', 'age', 'side']]

        for field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False
    except (ValueError, KeyError, TypeError) as e:
        return False

    return True


def convert_csv_row_types(row: dict) -> dict:
    try:
        return {
            'sex': int(row['sex']),
            'age': int(row['age']),
            'side': int(row['side']),
            'BW': float(row['BW']),
            'Ht': float(row['Ht']),
            'BMI': float(row['BMI']),
            'IKDC_pre': float(row['IKDC pre']),
            'Lysholm_pre': float(row['Lysholm pre']),
            'Pre_KL_grade': float(row['Pre KL grade']),
            'MM_extrusion_pre': float(row['MM extrusion pre']),
        }
        
    except ValueError:
        return {}
