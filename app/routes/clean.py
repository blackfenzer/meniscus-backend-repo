from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from requests import Session
from app.handlers.clean_handler import clean_train_request_csv
from app.database.session import get_db
from app.models.schema import CSVFile
from loguru import logger

router = APIRouter()


@router.post("/clean-and-upload")
async def clean_and_upload_csv(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")

    try:
        # Read CSV content
        csv_content = (await file.read()).decode("utf-8")

        # Clean CSV
        cleaned_csv = clean_train_request_csv(csv_content)
        if cleaned_csv is None:
            raise HTTPException(status_code=500, detail="Error cleaning CSV")

        # Save cleaned CSV to the same DB as validate_and_upload_csv
        CSVFile.create_from_csv(db, cleaned_csv)

        return {"valid": True, "message": "CSV cleaned and uploaded successfully"}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")
