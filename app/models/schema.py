from datetime import datetime
from sqlalchemy import Column, DateTime, Double, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship
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

class CSVFile(Base):
    __tablename__ = "csvFiles"
    id = Column(Integer, primary_key=True)
    last_modified_time = Column(DateTime, default=datetime.utcnow)
    model_architecture = Column(String)

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    model_architecture = Column(String)
    train_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    final_loss = Column(Double)

class Sent(Base):
    __tablename__ = "sents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    csv_file_id = Column(Integer, ForeignKey("csvFiles.id"), nullable=False)
    sent_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("Users", back_populates="sents")
    csv_file = relationship("CSVFile", back_populates="sents")