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


@router.get("/csrf-token")
def get_csrf_token(csrf_protect: CsrfProtect = Depends()):
    # Generate a new CSRF token
    csrf_token = csrf_protect.generate_csrf()
    return {"csrf_token": csrf_token}


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@router.post("/token")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # 🔹 Add password hashing check here (e.g., bcrypt.checkpw)

    return {"access_token": "dummy_token"}


@router.get("/check-login")
def check_login(token: str = Depends(oauth2_scheme)):
    payload = validate_nextauth_jwt(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"message": "User is logged in", "username": payload.get("sub")}


@router.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=username, password=password)
    db.add(new_user)
    db.commit()

    return {"message": "User created"}
