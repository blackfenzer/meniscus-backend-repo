from pydantic import BaseModel, Field
from pydantic import ConfigDict
from typing import Optional
from datetime import datetime


class UserSchema(BaseModel):
    username: str
    password: str
    is_admin: bool
    is_active: bool


class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserSchema


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False  # Add is_admin field


class PredictRequest(BaseModel):
    sex: int
    age: int
    side: int
    BW: float
    Ht: float
    BMI: float
    IKDC_pre: float = Field(..., alias="IKDC pre")
    Lysholm_pre: float = Field(..., alias="Lysholm pre")
    Pre_KL_grade: float = Field(..., alias="Pre KL grade")
    MM_extrusion_pre: float = Field(..., alias="MM extrusion pre")


class TrainConfig(BaseModel):
    learning_rate: float = Field(
        ..., gt=0, description="Learning rate for the optimizer"
    )
    batch_size: int = Field(
        ..., gt=0, description="Number of samples per training batch"
    )
    num_epochs: int = Field(..., gt=0, description="Number of training epochs")
    dataset_path: str = Field(..., description="File path to the training dataset")
    model_name: Optional[str] = Field(None, description="Optional name for the model")


class ModelResponse(BaseModel):
    model_id: str
    status: str
    accuracy: Optional[float] = None
    message: Optional[str] = None
