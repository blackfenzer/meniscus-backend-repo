import os
from fastapi import APIRouter, Depends, Response, Request, HTTPException
from fastapi.security import OAuth2PasswordBearer
from itsdangerous import URLSafeTimedSerializer
from pydantic import BaseModel
from datetime import timedelta, datetime
import secrets
from jose import JWTError, jwt
from app.database.session import get_db
from app.models.schema import User
from sqlalchemy.orm import Session

router = APIRouter()

# Secret keys for signing cookies & CSRF tokens
SECRET_KEY = os.getenv("SECRET_KEY")
CSRF_SECRET = os.getenv("CSRF_SECRET")
COOKIE_NAME = os.getenv("COOKIE_NAME")
CSRF_COOKIE_NAME = os.getenv("CSRF_COOKIE_NAME")

SECRET_KEY = os.getenv("SECRET_KEY")  # Use environment variable in production
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Serializer for signed cookies
serializer = URLSafeTimedSerializer(SECRET_KEY)

# Role-based CORS configuration
origins = {
    "admin": ["https://admin.example.com"],
    "user": ["https://user.example.com"],
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


# Utility functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError as e:
        return None


def role_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        role = payload.get("role")
        if role is None:
            return None
        return role
    except JWTError:
        return None


async def get_token(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    return token


# Dependency to get current user
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


async def protected_route(request: Request, db: Session = Depends(get_db)):
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
    if not user.role == "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    return user


async def get_current_role(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    role = role_token(token)
    if not role:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return role


async def get_current_token(request: Request, db: Session = Depends(get_db)):
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return token


async def check_user():
    try:
        if Depends(get_current_user) == "user":
            return True
        return False
    except:
        return False


async def check_admin():
    try:
        if Depends(get_current_user) == "admin":
            return True
        return False
    except:
        return False


# Routes
@router.post("/login")
async def login(
    request: LoginRequest, response: Response, db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not user.check_password(request.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create access token
    access_token = create_access_token({"sub": user.username, "role": user.role})

    # Generate CSRF token
    csrf_token = secrets.token_urlsafe(32)

    # Set HTTP-only cookie for JWT token
    response.set_cookie(
        COOKIE_NAME,
        access_token,
        httponly=True,  # Make cookie HTTP-only
        secure=True,  # Only send over HTTPS
        samesite="lax",  # Protect against CSRF
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
    )

    # Set CSRF token cookie (not HTTP-only so JavaScript can read it)
    response.set_cookie(
        "csrf_token",
        csrf_token,
        httponly=False,
        secure=True,
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

    return {
        "message": "Login successful",
        "access_token": access_token,
        "csrf_token": csrf_token,
    }


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    response.delete_cookie("csrf_token")
    return {"message": "Logged out successfully"}


@router.get("/me")
async def test_auth(current_user: User = Depends(get_current_user)):
    return {
        "message": f"Hello, {current_user.username}! You are authenticated.",
        "username": current_user.username,
        "role": current_user.role,
    }


def is_valid_password(password: str) -> bool:
    """Check if password is at least 8 characters long, contains at least one number and one special character."""
    return (
        len(password) >= 8
        and any(char.isdigit() for char in password)
        and any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~" for char in password)
    )


@router.post("/register")
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    if not is_valid_password(request.password):
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long, contain at least one number and one special character",
        )

    user = db.query(User).filter(User.username == request.username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=request.username, role="user")
    new_user.set_password(request.password)  # Hash the password
    db.add(new_user)
    db.commit()

    return {"message": "User created"}
