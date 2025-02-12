import csv
from datetime import datetime
from requests import Session
from sqlalchemy import Column, DateTime, Double, ForeignKey, Integer, String, Boolean
from sqlalchemy.orm import DeclarativeBase, relationship
import bcrypt
import sqlalchemy

from backend.app.database.session import SessionLocal

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
    file_path = Column(String, nullable=False)
    last_modified_time = Column(DateTime, default=datetime.utcnow)
    model_architecture = Column(String)

    def load_csv_to_db(file_path: str):
        db: Session = SessionLocal()
        try:
            csv_file = db.query(CSVFile).filter(CSVFile.file_path == file_path).first()
            if not csv_file:
                print(f"No CSVFile record found for file_path: {file_path}")
                return

            with open(file_path, mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    data_entry = Data(
                        csv_file_id = csv_file.id,
                        sex = int(row['sex']) if row['sex'] else None,
                        age = int(row['age']) if row['age'] else None,
                        side = float(row['side']) if row['side'] else None,
                        BW = float(row['BW']) if row['BW'] else None,
                        Ht = float(row['Ht']) if row['Ht'] else None,
                        BMI = float(row['BMI']) if row['BMI'] else None,
                        IKDC_pre = float(row['IKDC pre']) if row['IKDC_pre'] else None,
                        IKDC_3_m = float(row['IKDC 3 m']) if row['IKDC 3 m'] else None,
                        IKDC_6_m = float(row['IKDC 6 m']) if row['IKDC 6 m'] else None,
                        IKDC_1_Y = float(row['IKDC 1 Y']) if row['IKDC 1 Y'] else None,
                        IKDC_2_Y = float(row['IKDC 2 Y']) if row['IKDC 2 Y'] else None,
                        Lysholm_pre = float(row['Lysholm pre']) if row['Lysholm pre'] else None,
                        Lysholm_3_m = float(row['Lysholm 3 m']) if row['Lysholm 3 m'] else None,
                        Lysholm_6_m = float(row['Lysholm 6 m']) if row['Lysholm 6 m'] else None,
                        Lysholm_1_Y = float(row['Lysholm 1 Y']) if row['Lysholm 1 Y'] else None,
                        Lysholm_2_Y = float(row['Lysholm 2 Y']) if row['Lysholm 2 Y'] else None,
                        Pre_KL_grade = float(row['Pre KL grade']) if row['Pre KL grade'] else None,
                        Post_KL_grade_2_Y = float(row['Post_KL_grade_2_Y']) if row['Post_KL_grade_2_Y'] else None,
                        MM_extrusion_pre = float(row['MM extrusion pre']) if row['MM extrusion pre'] else None,
                        MM_extrusion_post = float(row['MM extrusion post']) if row['MM extrusion post'] else None
                    )
                    db.add(data_entry)
            db.commit()
            print(f"Data from {file_path} has been successfully loaded into the database.")

        except Exception as e:
            db.rollback()
            print(f"An error occurred: {e}")

        finally:
            db.close()

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

class Data(Base):
    __tablename__ = "data"
    csv_file_id = Column(Integer, ForeignKey("csvFiles.id"), nullable=False)
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

    csv_file = relationship("CSVFile", back_populates="sents")