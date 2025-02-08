from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship
import bcrypt
import sqlalchemy

Base = sqlalchemy.orm.declarative_base()

class Base(DeclarativeBase):
    pass

class Sent(Base):
    __tablename__ = "sents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user.id"), nullable=False)
    csv_file_id = Column(Integer, ForeignKey("csvFile.id"), nullable=False)
    sent_date = Column(DateTime, default=datetime.utcnow, nullable=False)