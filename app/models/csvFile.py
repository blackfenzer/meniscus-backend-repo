from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase
import bcrypt
import sqlalchemy

Base = sqlalchemy.orm.declarative_base()

class Base(DeclarativeBase):
    pass

class CSVFile(Base):
    __tablename__ = "csv files"
    id = Column(Integer, primary_key=True)
    last_modified_time = Column(DateTime, default=datetime.utcnow)
    model_architecture = Column(String)