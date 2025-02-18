from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from sqlalchemy.orm import Session
from datetime import datetime
import csv
from io import StringIO
from app.database.session import get_db
from app.models.schema import CSVFile, CSVData
from app.schemas.schemas import CSVFileResponse

router = APIRouter()


# Create a CSV file entry and store data
@router.post("/", response_model=CSVFileResponse)
async def upload_csv_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        contents = await file.read()
        csv_content = contents.decode("utf-8")
        csv_file = CSVFile.create_from_csv(db, csv_content)
        return csv_file
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing CSV file: {str(e)}"
        )


# Read all CSV files
@router.get("/", response_model=list[CSVFileResponse])
def read_csv_files(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(CSVFile).offset(skip).limit(limit).all()


# Read a specific CSV file by ID
@router.get("/{csv_file_id}", response_model=CSVFileResponse)
def read_csv_file(csv_file_id: int, db: Session = Depends(get_db)):
    csv_file = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
    return csv_file


# Update CSV file metadata
@router.put("/{csv_file_id}", response_model=CSVFileResponse)
def update_csv_file(
    csv_file_id: int, model_architecture: str, db: Session = Depends(get_db)
):
    csv_file = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file:
        raise HTTPException(status_code=404, detail="CSV file not found")

    csv_file.model_architecture = model_architecture
    csv_file.last_modified_time = datetime.now()

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
