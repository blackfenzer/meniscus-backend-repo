from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.database.session import get_db
from app.models.schema import User
from app.schemas.schemas import (
    UserSchema,
    UserUpdateSchema,
)
from app.routes.auth2 import get_current_user, protected_route
from loguru import logger

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 30
router = APIRouter()


@router.get("/", response_model=list[UserSchema])
def all_user(
    db: Session = Depends(get_db), user: UserSchema = Depends(protected_route)
):
    users = db.query(User)
    return users


@router.get("/{user_id}", response_model=UserSchema)
def read_user(
    user_id: int,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(protected_route),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def is_valid_password(password: str) -> bool:
    """Check if password is at least 8 characters long, contains at least one number and one special character."""
    return (
        len(password) >= 8
        and any(char.isdigit() for char in password)
        and any(char in "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~" for char in password)
    )


@router.put("/{user_id}", response_model=UserUpdateSchema)
def update_user(
    user_id: int,
    user: UserUpdateSchema,
    db: Session = Depends(get_db),
    current_user: UserSchema = Depends(get_current_user),
):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")

    user_data = user.model_dump(exclude_unset=True)
    if current_user.role != "admin":
        # Only allow normal users to update username and password.
        user_data.pop("role", None)
        user_data.pop("is_active", None)

    for key, value in user_data.items():
        if value is not None and value != "":
            if key is "password":
                if not is_valid_password(value):
                    raise HTTPException(
                        status_code=400,
                        detail="Password must be at least 8 characters long, contain at least one number and one special character",
                    )
                db_user.set_password(value)
            else:
                setattr(db_user, key, value)

    try:
        db.commit()
        db.refresh(db_user)
    except Exception as e:
        db.rollback()
        logger.error(f"Internal error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    return db_user


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    user: UserSchema = Depends(protected_route),
):
    db_user = db.query(User).filter(User.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()
    return {"message": "User deleted successfully"}
