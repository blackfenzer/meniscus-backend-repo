from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase
import bcrypt
import sqlalchemy

Base = sqlalchemy.orm.declarative_base()

class Base(DeclarativeBase):
    pass
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # Store hashed password
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user")

    def set_password(self, password: str):
        # Hash the password
        self.password = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def check_password(self, password: str) -> bool:
        # Verify the password
        return bcrypt.checkpw(password.encode("utf-8"), self.password.encode("utf-8"))
