from pydantic import BaseModel
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
