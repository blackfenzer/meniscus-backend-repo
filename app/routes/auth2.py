from fastapi import APIRouter, FastAPI, Depends, Response, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from itsdangerous import URLSafeTimedSerializer, BadSignature
from starlette.responses import JSONResponse
from datetime import timedelta, datetime
import secrets
from jose import JWTError, jwt
from app.database.session import get_db
from app.models.schema import User
from sqlalchemy.orm import Session

router = APIRouter()
# security = HTTPBearer()

# Secret keys for signing cookies & CSRF tokens
SECRET_KEY = "super-secret-key-change-this"
CSRF_SECRET = "csrf-secret-key-change-this"
COOKIE_NAME = "session_token"
CSRF_COOKIE_NAME = "csrf_token"

SECRET_KEY = "your-secret-key-change-this"  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
# Serializer for signed cookies
serializer = URLSafeTimedSerializer(SECRET_KEY)

# Role-based CORS configuration
origins = {
    "admin": ["https://admin.example.com"],
    "user": ["https://user.example.com"],
}

# Dummy user data
# users_db = {
#     "admin": {"username": "admin", "password": "adminpass", "role": "admin"},
#     "user": {"username": "user", "password": "userpass", "role": "user"},
# }

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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
    except JWTError:
        return None


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


# Routes
@router.post("/login")
async def login(
    response: Response, username: str, password: str, db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.username == username).first()
    if not user or not user.check_password(password):
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


@router.post("/submit-data")
async def submit_data(request: Request, current_user: User = Depends(get_current_user)):
    csrf_token = request.cookies.get("csrf_token")
    header_csrf_token = request.headers.get("X-CSRF-Token")
    if csrf_token:
        print(csrf_token)
    if header_csrf_token:
        print(header_csrf_token)
    if not csrf_token or not header_csrf_token or csrf_token != header_csrf_token:
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

    # data = await request.json()
    # Process the data here
    return {"message": "Data submitted successfully"}


# @router.post("/submit-data")
# async def submit_data(request: Request, user: dict = Depends(get_current_user)):
#     csrf_token = request.cookies.get(CSRF_COOKIE_NAME)
#     header_csrf_token = request.headers.get("X-CSRF-Token")

#     if not csrf_token or not header_csrf_token or csrf_token != header_csrf_token:
#         raise HTTPException(status_code=403, detail="CSRF token missing or invalid")

#     data = await request.json()
#     # Process the data here
#     return {"message": "Data submitted successfully"}


@router.post("/register")
def register(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already exists")

    new_user = User(username=username, role="user")
    new_user.set_password(password)  # Hash the password
    db.add(new_user)
    db.commit()

    return {"message": "User created"}
