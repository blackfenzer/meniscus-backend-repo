import os
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime
import csv
from io import StringIO
from app.database.session import get_db
from app.models.schema import CSVFile, CSVData
from app.schemas.schemas import (
    CSVFileBase,
    CSVFileListResponse,
    CSVFileResponse,
    CSVUpdate,
)
from app.handlers.validate_handler import validate_train_request_csv

router = APIRouter()


# Create a CSV file entry and store data
@router.post("/upload")
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


# Read all CSV files
@router.get("/all", response_model=CSVFileListResponse)
async def list_csv_files(db: Session = Depends(get_db)):
    csv_files = db.query(CSVFile).all()
    return {
        "files": [
            CSVFileBase(
                id=csv_file.id,
                last_modified_time=csv_file.last_modified_time,
                model_architecture=csv_file.model_architecture,
                length=csv_file.length,
            )
            for csv_file in csv_files
        ]
    }


# Read a specific CSV file by ID
@router.get("/{csv_file_id}", response_model=CSVFileBase)
def read_csv_file(csv_file_id: int, db: Session = Depends(get_db)):
    csv_file = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
    return csv_file


# Update CSV file metadata
@router.put("/{csv_file_id}", response_model=CSVFileBase)
def update_csv_file(csv_file_id: int, model: CSVUpdate, db: Session = Depends(get_db)):
    csv_file = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    for key, value in model.dict(exclude_unset=True).items():
        if value is not None and value != "":
            setattr(csv_file, key, value)

    try:
        db.commit()
        db.refresh(csv_file)
        return csv_file
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")


# Delete a CSV file entry
@router.delete("/{csv_file_id}")
def delete_csv_file(csv_file_id: int, db: Session = Depends(get_db)):
    csv_file = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    try:
        # Delete related data entries
        db.query(CSVData).filter(CSVData.csv_file_id == csv_file.id).delete()

        # Delete CSV file record
        db.delete(csv_file)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, detail=f"Database deletion failed: {str(e)}"
        )

    return {"message": "CSV file deleted successfully"}
