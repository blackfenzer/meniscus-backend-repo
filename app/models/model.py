from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Double
from sqlalchemy.orm import DeclarativeBase
import bcrypt
import sqlalchemy

Base = sqlalchemy.orm.declarative_base()

class Base(DeclarativeBase):
    pass

class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    model_architecture = Column(String)
    train_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    final_loss = Column(Double)