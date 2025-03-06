from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from app.database.session import get_db
from app.models.schema import User
from app.schemas.schemas import (
    LoginRequest,
    RegisterRequest,
    Token,
    CSVFileResponse,
    UserSchema,
    UserUpdateSchema,
)
from app.security.jwt_handler import create_access_token, validate_nextauth_jwt

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30
router = APIRouter()


@router.get("/", response_model=list[UserSchema])
def read_user(db: Session = Depends(get_db)):
    users = db.query(User)
    return users


@router.get("/{user_id}", response_model=UserSchema)
def read_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

def is_valid_password(password: str) -> bool:
    """Check if password is at least 8 characters long, contains at least one number and one special character."""
    return (
        len(password) >= 8 and
        any(char.isdigit() for char in password) and
        any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~" for char in password)
    )

@router.put("/{user_id}", response_model=UserUpdateSchema)
def update_user(user_id: int, user: UserUpdateSchema, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    user_data = user.model_dump(exclude_unset=True)
    for key, value in user_data.items():
        if value is not None and value != "":
            if key is "password":
                if not is_valid_password(value):
                    raise HTTPException(
                        status_code=400, 
                        detail="Password must be at least 8 characters long, contain at least one number and one special character"
                    )
                db_user.set_password(value)
            else:
                setattr(db_user, key, value)

    try:
        db.commit()
        db.refresh(db_user)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    return db_user


@router.delete("/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return {"message": "User deleted successfully"}
