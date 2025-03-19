from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPBearer
from app.security.csrf_handler import CsrfProtect
from typing import Annotated
import os
from fastapi_nextauth_jwt import NextAuthJWT

router = APIRouter()
security = HTTPBearer()
JWT = NextAuthJWT(
    secret=os.getenv("NEXTAUTH_SECRET"),
)


@router.post("/protected")
def protected_route(
    request: Request,
    jwt: Annotated[dict, Depends(JWT)],
    csrf_protect: CsrfProtect = Depends(),
):
    # Validate CSRF Token
    csrf_protect.validate_csrf(request)

    # Validate JWT
    payload = jwt["name"]
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {"message": "Access granted", "user": payload}


