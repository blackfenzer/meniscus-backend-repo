import os
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import csv
from io import StringIO
from app.database.session import get_db
from app.models.schema import CSVFile, CSVData
from app.schemas.schemas import (
    CSVFileBase,
    CSVFileListResponse,
    CSVUpdate,
    UserSchema,
)
from app.handlers.validate_handler import validate_train_request_csv
from app.routes.auth2 import get_current_user, protected_route
from loguru import logger

router = APIRouter()


# Create a CSV file entry and store data
@router.post("/upload")
async def validate_and_upload_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: UserSchema = Depends(get_current_user),
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
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


# Read all CSV files
@router.get("/all", response_model=CSVFileListResponse)
async def list_csv_files(
    db: Session = Depends(get_db),
    user: UserSchema = Depends(get_current_user),
):
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
def read_csv_file(
    csv_file_id: int,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(get_current_user),
):
    csv_file = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file:
        raise HTTPException(status_code=404, detail="CSV file not found")
    return csv_file


# Update CSV file metadata
@router.put("/{csv_file_id}", response_model=CSVFileBase)
def update_csv_file(
    csv_file_id: int,
    model: CSVUpdate,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(protected_route),
):
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
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database update failed: {str(e)}")


# Delete a CSV file entry
@router.delete("/{csv_file_id}")
def delete_csv_file(
    csv_file_id: int,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(protected_route),
):
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
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database deletion failed: {str(e)}"
        )

    return {"message": "CSV file deleted successfully"}


@router.get("/download/{csv_file_id}")
def download_csv(
    csv_file_id: int,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(get_current_user),
):
    # Retrieve the CSVFile record; adjust filtering as needed
    csv_file_record = db.query(CSVFile).filter(CSVFile.id == csv_file_id).first()
    if not csv_file_record:
        raise HTTPException(status_code=404, detail="CSV file not found")

    # Retrieve associated data entries
    data_entries = db.query(CSVData).filter(CSVData.csv_file_id == csv_file_id).all()

    # Create an in-memory CSV file
    output = StringIO()
    writer = csv.writer(output)

    # Write the header row (adjust the headers to match your Data model)
    headers = [
        "sex",
        "age",
        "side",
        "BW",
        "Ht",
        "BMI",
        "IKDC pre",
        "IKDC 3 m",
        "IKDC 6 m",
        "IKDC 1 Y",
        "IKDC 2 Y",
        "Lysholm pre",
        "Lysholm 3 m",
        "Lysholm 6 m",
        "Lysholm 1 Y",
        "Lysholm 2 Y",
        "Pre KL grade",
        "Post KL grade 2 Y",
        "MM extrusion pre",
        "MM extrusion post",
        "MM gap",
        "Degenerative meniscus",
        "medial femoral condyle",
        "medial tibial condyle",
        "lateral femoral condyle",
        "lateral tibial condyle",
    ]
    writer.writerow(headers)

    # Write each data row
    for entry in data_entries:
        writer.writerow(
            [
                entry.sex,
                entry.age,
                entry.side,
                entry.BW,
                entry.Ht,
                entry.BMI,
                entry.IKDC_pre,
                entry.IKDC_3_m,
                entry.IKDC_6_m,
                entry.IKDC_1_Y,
                entry.IKDC_2_Y,
                entry.Lysholm_pre,
                entry.Lysholm_3_m,
                entry.Lysholm_6_m,
                entry.Lysholm_1_Y,
                entry.Lysholm_2_Y,
                entry.Pre_KL_grade,
                entry.Post_KL_grade_2_Y,
                entry.MM_extrusion_pre,
                entry.MM_extrusion_post,
                entry.MM_gap,
                entry.Degenerative_meniscus,
                entry.medial_femoral_condyle,
                entry.medial_tibial_condyle,
                entry.lateral_femoral_condyle,
                entry.lateral_tibial_condyle,
            ]
        )

    # Reset the StringIO object's cursor to the beginning
    output.seek(0)

    # Create a StreamingResponse to send the CSV file
    headers = {"Content-Disposition": "attachment; filename=export.csv"}
    return StreamingResponse(output, media_type="text/csv", headers=headers)
