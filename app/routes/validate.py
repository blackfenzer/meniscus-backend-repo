import base64
import os
from fastapi.responses import StreamingResponse
import httpx
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Request
from app.models.schema import User

from sqlalchemy.orm import Session
from app.handlers.clean_handler import clean_train_request_csv
from app.database.session import get_db
from app.models.schema import CSVFile, CSVData, Model
from io import StringIO
import csv

from jose import JWTError, jwt
import numpy as np
from loguru import logger

router = APIRouter()
COOKIE_NAME = os.getenv("COOKIE_NAME")
CSRF_COOKIE_NAME = os.getenv("CSRF_COOKIE_NAME")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None


async def get_current_user(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    username = verify_token(token)
    if not username:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user


HOST = os.getenv("BENTOML_HOST")
BENTOML_URL = f"{HOST}train_model"

@router.post("/model_train")
async def model_train_endpoint(
    name: str,
    version: str,
    description: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    logger.error(f"Host: {HOST}")
    logger.error(f"Bentoml URL: {BENTOML_URL}")

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

        csv_base64 = base64.b64encode(csv_bytes).decode("utf-8")
        
        token_data = {
        "user_id": str(user.id),
        "role": str(user.role),
        }
        secure_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        
        payload = {
            "name": name,
            "version": version,
            "description": description,
            "csv_bytes": csv_base64,
            "model_params": {"hidden_dim": 251, "num_layers": 8, "dropout": 0.1},
            "optimizer_params": {"lr": 0.01905, "weight_decay": 0.01754},
            "batch_size": 32,
            "epochs": 100,
            "secure_token": secure_token,  # Pass your secure token here
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BENTOML_URL}",
                json={"payload": payload},
                timeout=30,
            )
        logger.error(f"Bentoml URL: {BENTOML_URL}")
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"BentoML service error: {response.text}")

        bentoml_response = response.json()

        # Save model record in DB
        db_model = Model(
            name=name,
            model_architecture="RegressionNet",
            model_path=file.filename,
            bentoml_tag=bentoml_response["bentoml_tag"],
            is_active=True,
            csv_id=csv_record.id,
            version=version,
            description=description,
            final_loss=bentoml_response.get("rmse"),
            r2=bentoml_response.get("r2"),
        )
        db.add(db_model)
        db.commit()

        return {"status": "success", "bentoml_tag": bentoml_response["bentoml_tag"]}

    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {str(e)}")


@router.get("/download/csv/{csv_file_id}")
def download_csv(csv_file_id: int, db: Session = Depends(get_db)):
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
