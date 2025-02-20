import csv
from datetime import datetime
from typing import List, Optional
from requests import Session
from sqlalchemy import (
    BLOB,
    Column,
    DateTime,
    Double,
    ForeignKey,
    Integer,
    String,
    Boolean,
)
from sqlalchemy.orm import DeclarativeBase, relationship
import bcrypt
import sqlalchemy
from io import StringIO
from app.database.session import SessionLocal

# Base = sqlalchemy.orm.declarative_base()


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)  # Store hashed password
    is_active = Column(Boolean, default=True)
    role = Column(String, default="user")

    sents = relationship("Sent", back_populates="user")

    def set_password(self, password: str):
        # Hash the password
        self.password = bcrypt.hashpw(
            password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

    def check_password(self, password: str) -> bool:
        # Verify the password
        return bcrypt.checkpw(password.encode("utf-8"), self.password.encode("utf-8"))


class CSVFile(Base):
    __tablename__ = "csv_files"
    id = Column(Integer, primary_key=True)
    last_modified_time = Column(DateTime, default=datetime.now())
    model_architecture = Column(String)
    length = Column(Integer)

    data_entries = relationship("CSVData", back_populates="csv_file")
    sents = relationship("Sent", back_populates="csv_file")

    @classmethod
    def create_from_csv(cls, db: Session, csv_content: str):
        # Create a new CSVFile entry
        csv_file = cls()
        db.add(csv_file)

        # Read CSV content from memory
        reader = csv.DictReader(StringIO(csv_content))

        rows = list(reader)
        length = len(rows)
        csv_file.length = length
        db.commit()
        db.refresh(csv_file)

        for row in rows:
            data_entry = CSVData(
                csv_file_id=csv_file.id,
                sex=int(row["sex"]) if row["sex"] else None,
                age=int(row["age"]) if row["age"] else None,
                side=float(row["side"]) if row["side"] else None,
                BW=float(row["BW"]) if row["BW"] else None,
                Ht=float(row["Ht"]) if row["Ht"] else None,
                BMI=float(row["BMI"]) if row["BMI"] else None,
                IKDC_pre=float(row["IKDC pre"]) if row.get("IKDC pre") else None,
                IKDC_3_m=float(row["IKDC 3 m"]) if row.get("IKDC 3 m") else None,
                IKDC_6_m=float(row["IKDC 6 m"]) if row.get("IKDC 6 m") else None,
                IKDC_1_Y=float(row["IKDC 1 Y"]) if row.get("IKDC 1 Y") else None,
                IKDC_2_Y=float(row["IKDC 2 Y"]) if row.get("IKDC 2 Y") else None,
                Lysholm_pre=(
                    float(row["Lysholm pre"]) if row.get("Lysholm pre") else None
                ),
                Lysholm_3_m=(
                    float(row["Lysholm 3 m"]) if row.get("Lysholm 3 m") else None
                ),
                Lysholm_6_m=(
                    float(row["Lysholm 6 m"]) if row.get("Lysholm 6 m") else None
                ),
                Lysholm_1_Y=(
                    float(row["Lysholm 1 Y"]) if row.get("Lysholm 1 Y") else None
                ),
                Lysholm_2_Y=(
                    float(row["Lysholm 2 Y"]) if row.get("Lysholm 2 Y") else None
                ),
                Pre_KL_grade=(
                    float(row["Pre KL grade"]) if row.get("Pre KL grade") else None
                ),
                Post_KL_grade_2_Y=(
                    float(row["Post_KL_grade_2_Y"])
                    if row.get("Post_KL_grade_2_Y")
                    else None
                ),
                MM_extrusion_pre=(
                    float(row["MM extrusion pre"])
                    if row.get("MM extrusion pre")
                    else None
                ),
                MM_extrusion_post=(
                    float(row["MM extrusion post"])
                    if row.get("MM extrusion post")
                    else None
                ),
            )
            db.add(data_entry)
        db.commit()


class Model(Base):
    __tablename__ = "models"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    model_architecture = Column(String)
    # train_date = Column(DateTime, default=datetime.now(), nullable=False)
    final_loss = Column(Double)
    model_path = Column(String)
    bentoml_tag = Column(String)
    # model_data = Column(BLOB)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now())
    csv_id = Column(Integer, ForeignKey("csv_files.id"), nullable=True)
    version = Column(String)
    description = Column(String)


class Sent(Base):
    __tablename__ = "sents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    csv_file_id = Column(Integer, ForeignKey("csv_files.id"), nullable=False)
    sent_date = Column(DateTime, default=datetime.now(), nullable=False)

    # Fix the relationships - use the class names, not table names
    user = relationship("User", back_populates="sents")
    csv_file = relationship("CSVFile", back_populates="sents")


class CSVData(Base):
    __tablename__ = "csv_data"
    id = Column(Integer, primary_key=True)

    sex = Column(Integer)
    age = Column(Integer)
    side = Column(Double)
    BW = Column(Double)
    Ht = Column(Double)
    BMI = Column(Double)
    IKDC_pre = Column(Double)
    IKDC_3_m = Column(Double)
    IKDC_6_m = Column(Double)
    IKDC_1_Y = Column(Double)
    IKDC_2_Y = Column(Double)
    Lysholm_pre = Column(Double)
    Lysholm_3_m = Column(Double)
    Lysholm_6_m = Column(Double)
    Lysholm_1_Y = Column(Double)
    Lysholm_2_Y = Column(Double)
    Pre_KL_grade = Column(Double)
    Post_KL_grade_2_Y = Column(Double)
    MM_extrusion_pre = Column(Double)
    MM_extrusion_post = Column(Double)

    csv_file_id = Column(Integer, ForeignKey("csv_files.id"), nullable=False)
    csv_file = relationship("CSVFile", back_populates="data_entries")
