from fastapi import APIRouter, Depends
from fastapi_csrf_protect import CsrfProtect
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from app.security.jwt_handler import validate_nextauth_jwt
from app.database.session import get_db
from app.models.user import User
from sqlalchemy.orm import Session
from pydantic import BaseModel

router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    is_admin: bool = False  # Add is_admin field


@router.get("/csrf-token")
def get_csrf_token(csrf_protect: CsrfProtect = Depends()):
    # Generate a new CSRF token
    csrf_token = csrf_protect.generate_csrf()
    return {"csrf_token": csrf_token}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@router.get("/check-login")
def check_login(token: str = Depends(oauth2_scheme)):
    payload = validate_nextauth_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"message": "User is logged in", "username": payload.get("sub")}


@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=request.username, is_admin=request.is_admin)
    new_user.set_password(request.password)  # Hash the password
    db.add(new_user)
    db.commit()

    return {"message": "User created"}


@router.post("/token")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not user.check_password(request.password):  # Verify password
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Return JWT token with user info
    return {
        "access_token": "dummy_token",
        "username": user.username,
        "is_admin": user.is_admin,  # Include is_admin in the response
    }
