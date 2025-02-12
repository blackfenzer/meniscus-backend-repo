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


class TrainRequest(BaseModel):
    sex: int
    age: int
    side: int
    BW: float
    Ht: float
    BMI: float
    IKDC_pre: float = Field(..., alias="IKDC pre")
    IKDC_3_m: float = Field(..., alias="IKDC 3 m")
    IKDC_6_m: float = Field(..., alias="IKDC 6 m")
    IKDC_1_Y: float = Field(..., alias="IKDC 1 Y")
    IKDC_2_Y: float = Field(..., alias="IKDC 2 Y");
    Lysholm_pre: float = Field(..., alias="Lysholm pre")
    Lysholm_3_m: float = Field(..., alias="Lysholm 3 m")
    Lysholm_6_m: float = Field(..., alias="Lysholm 6 m")
    Lysholm_1_Y: float = Field(..., alias="Lysholm 1 Y")
    Lysholm_2_Y: float = Field(..., alias="Lysholm 2 Y")
    Pre_KL_grade: float = Field(..., alias="Pre KL grade")
    Post_KL_grade_2_Y: float = Field(..., alias="Post KL grade 2 Y")
    MRI_healing_1_Y: float = Field(..., alias="MRI healing 1 Y")
    MM_extrusion_pre: float = Field(..., alias="MM extrusion pre")
    MM_extrusion_post: float = Field(..., alias="MMextrusion post")


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
