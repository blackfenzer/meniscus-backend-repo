from datetime import timedelta
from fastapi import APIRouter, Depends
from fastapi_csrf_protect import CsrfProtect
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from app.security.jwt_handler import create_access_token, validate_nextauth_jwt
from app.database.session import get_db
from app.models.user import User
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.schemas.schemas import RegisterRequest, LoginRequest, UserSchema, Token
from fastapi_nextauth_jwt import NextAuthJWT

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30
router = APIRouter()


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


# JWT = NextAuthJWT(
#     secret="y0uR_SuP3r_s3cr37_$3cr3t",
# )


@router.post("/token")
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not user.check_password(request.password):  # Verify password
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    # access_token = create_access_token(
    #     data={"sub": str(user.id)}, expires_delta=access_token_expires
    # )
    access_token = create_access_token({"sub": user.username} , expires_delta=access_token_expires)  # Assuming you have this function

    # Convert SQLAlchemy User instance to Pydantic UserSchema
    user_schema = UserSchema(
        username=user.username,
        password=user.password,  # Consider not returning the password for security
        is_admin=user.is_admin,
        is_active=user.is_active
    )

    return Token(access_token=access_token, token_type="Bearer", user=user_schema)
