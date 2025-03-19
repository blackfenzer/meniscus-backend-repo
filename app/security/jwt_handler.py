from datetime import datetime, timedelta, timezone
from fastapi_nextauth_jwt import NextAuthJWT
import os
from dotenv import load_dotenv
from jose import jwt

load_dotenv()

SECRET_KEY = os.getenv("NEXTAUTH_SECRET")
nextauth_jwt = NextAuthJWT(secret=SECRET_KEY)
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30
ACCESS_TOKEN_VALID_MINUTES = 1


def validate_nextauth_jwt(token: str):
    try:
        payload = nextauth_jwt.decode(token)
        return payload
    except Exception as e:
        print(f"JWT validation error: {e}")
        return None


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token
