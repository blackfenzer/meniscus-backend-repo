import logging
import os
from pydantic import BaseModel
from app.database.session import get_db
from app.routes.auth2 import get_token, protected_route, get_current_user
from sqlalchemy.orm import Session
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from app.models.schema import Model, User
from app.schemas.schemas import PredictRequest
from jose import jwt
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import httpx
from loguru import logger

router = APIRouter()
HOST = os.getenv("BENTOML_HOST")
BENTOML_URL = f"{HOST}predict"
BENTOML_URL_XG = f"{HOST}predictxg"

class PredictionRequest(BaseModel):
    model_tag: str
    input_data: PredictRequest


# @router.post("/upload")
# async def upload_model(
#     name: str,
#     version: str,
#     description: str,
#     file: UploadFile = File(...),
#     db: Session = Depends(get_db),
#     current_user: User = Depends(protected_route),
# ):
#     # Authentication check here (add your logic)
#     # if (current_user.role != "user"):
#     #     raise HTTPException(status_code=403, detail="User unauthorized")

#     # Load and verify model
#     try:
#         content = await file.read()
#         checkpoint = torch.load(
#             BytesIO(content), map_location="cpu", weights_only=False
#         )

#         # Initialize model with your architecture
#         model = RegressionNet(input_dim=10, hidden_dim=151, num_layers=2, dropout=0.15)
#         model.load_state_dict(checkpoint["model_state_dict"])
#         model.eval()
#         scripted_model = torch.jit.script(model)
#         # Save to BentoML model store
#         bento_model = bentoml.torchscript.save_model(
#             name,
#             scripted_model,
#             custom_objects={
#                 "scaler": checkpoint["scaler"],
#                 "config": {
#                     "input_dim": 10,
#                     "hidden_dim": 151,
#                     "num_layers": 2,
#                     "dropout": 0.15,
#                 },
#             },
#             labels={"version": version, "description": description},
#         )

#         # Store in database
#         db_model = Model(
#             name=name,
#             model_architecture="RegressionNet",
#             model_path=file.filename,
#             # model_data=content,
#             bentoml_tag=str(bento_model.tag),
#             is_active=True,
#         )
#         db.add(db_model)
#         db.commit()

#         return {"status": "success", "bentoml_tag": str(bento_model.tag)}

#     except Exception as e:
#         logger.error(f"Internal error: {str(e)}")
#         raise HTTPException(500, f"Model upload failed: {str(e)}")


SECRET_KEY = os.getenv("SECRET_KEY")  # Store this securely, ideally in environment variables
ALGORITHM = os.getenv("ALGORITHM")


@router.post("/{model_name}")
async def predict(
    model_name: str,
    prediction_request: PredictionRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # Extract data from the request
    input_data = prediction_request.input_data.dict()

    # Retrieve model metadata from the database
    model = (
        db.query(Model)
        .filter(Model.name == model_name, Model.is_active == True)
        .first()
    )

    if not model:
        raise HTTPException(status_code=404, detail="Model not found or inactive")

    token_data = {
        "user_id": str(user.id),
        "role": str(user.role),
    }
    secure_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    print("Secure token:", secure_token)

    # Forward the request to the BentoML service
    try:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(f"Received input data: {input_data}")

        async with httpx.AsyncClient() as client:
            if (model.model_architecture == "RegressionNet"):
                response = await client.post(
                    BENTOML_URL,
                    json={
                        "payload": {
                            "model_tag": model.bentoml_tag,
                            "input_data": input_data,
                            "secure_token": secure_token,
                        },
                        "headers": {"Authorization": f"Bearer {get_token}"},
                    },
                    timeout=30,
                )
            else:
                logger.info("enter")
                response = await client.post(
                    BENTOML_URL_XG,
                    json={
                        "payload": {
                            "model_tag": model.bentoml_tag,
                            "input_data": input_data,
                            "secure_token": secure_token,
                        },
                        "headers": {"Authorization": f"Bearer {get_token}"},
                    },
                    timeout=30,
                )
        response.raise_for_status()
        return response.json()

    except httpx.RequestError as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=502, detail=f"Prediction service unavailable: {str(e)}"
        )
    except ValueError as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(
            status_code=400, detail=f"JSON serialization error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


COOKIE_NAME = os.getenv("COOKIE_NAME")
