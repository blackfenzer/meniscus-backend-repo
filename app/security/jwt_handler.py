from fastapi_nextauth_jwt import NextAuthJWT
import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("NEXTAUTH_SECRET")
nextauth_jwt = NextAuthJWT(secret=SECRET_KEY)


def validate_nextauth_jwt(token: str):
    try:
        payload = nextauth_jwt.decode(token)
        return payload
    except Exception as e:
        print(f"JWT validation error: {e}")
        return None
