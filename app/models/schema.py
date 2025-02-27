import csv
from datetime import datetime
from typing import List, Optional
from requests import Session
from sqlalchemy import (
    ForeignKey,
    String,
    DateTime,
    Boolean,
    Double,
    Integer,
    BLOB,
    Float,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
import bcrypt
import sqlalchemy
from io import StringIO
from app.database.session import SessionLocal, Base

# Base = sqlalchemy.orm.declarative_base()


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    password: Mapped[str]  # Store hashed password
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    role: Mapped[str] = mapped_column(String(20), default="user")

    sents: Mapped[List["Sent"]] = relationship(back_populates="user")

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
    id: Mapped[int] = mapped_column(primary_key=True)
    last_modified_time: Mapped[datetime] = mapped_column(default=datetime.now)
    model_architecture: Mapped[Optional[str]] = mapped_column(String(50))
    length: Mapped[int] = mapped_column(Integer, default=0)

    data_entries: Mapped[List["CSVData"]] = relationship(back_populates="csv_file")
    sents: Mapped[List["Sent"]] = relationship(back_populates="csv_file")

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
                    float(row["Post KL grade 2 Y"])
                    if row.get("Post KL grade 2 Y")
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
                MM_gap=float(row["MM gap"]) if row.get("MM gap") else None,
                Degenerative_meniscus=
                    float(row["Degenerative meniscus"])
                    if row.get("Degenerative meniscus")
                    else None
                ,
                medial_femoral_condyle=
                    float(row["medial femoral condyle"])
                    if row.get("medial femoral condyle")
                    else None
                ,
                medial_tibial_condyle=
                    float(row["medial tibial condyle"])
                    if row.get("medial tibial condyle")
                    else None
                ,
                lateral_femoral_condyle=
                    float(row["lateral femoral condyle"])
                    if row.get("lateral femoral condyle")
                    else None
                ,
                lateral_tibial_condyle=
                    float(row["lateral tibial condyle"])
                    if row.get("lateral tibial condyle")
                    else None,
            )
            db.add(data_entry)
        db.commit()

        return csv_file


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    model_architecture: Mapped[str] = mapped_column(String(50))
    final_loss: Mapped[Optional[float]] = mapped_column(Float)
    r2: Mapped[Optional[float]] = mapped_column(Float)
    model_path: Mapped[Optional[str]] = mapped_column(String(100))
    bentoml_tag: Mapped[Optional[str]] = mapped_column(String(50))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)
    csv_id: Mapped[Optional[int]] = mapped_column(ForeignKey("csv_files.id"))
    version: Mapped[Optional[str]] = mapped_column(String(20))
    description: Mapped[Optional[str]] = mapped_column(String(200))


class Sent(Base):
    __tablename__ = "sents"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    csv_file_id: Mapped[int] = mapped_column(ForeignKey("csv_files.id"))
    sent_date: Mapped[datetime] = mapped_column(default=datetime.now)

    user: Mapped["User"] = relationship(back_populates="sents")
    csv_file: Mapped["CSVFile"] = relationship(back_populates="sents")


class CSVData(Base):
    __tablename__ = "csv_data"
    id: Mapped[int] = mapped_column(primary_key=True)
    sex: Mapped[Optional[int]]
    age: Mapped[Optional[int]]
    side: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    BW: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Ht: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    BMI: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    IKDC_pre: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    IKDC_3_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    IKDC_6_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    IKDC_1_Y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    IKDC_2_Y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Lysholm_pre: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Lysholm_3_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Lysholm_6_m: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Lysholm_1_Y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Lysholm_2_Y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Pre_KL_grade: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Post_KL_grade_2_Y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    MM_extrusion_pre: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    MM_extrusion_post: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    MM_gap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    Degenerative_meniscus: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    medial_femoral_condyle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    medial_tibial_condyle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lateral_femoral_condyle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lateral_tibial_condyle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    csv_file_id: Mapped[int] = mapped_column(ForeignKey("csv_files.id"))
    csv_file: Mapped["CSVFile"] = relationship(back_populates="data_entries")
