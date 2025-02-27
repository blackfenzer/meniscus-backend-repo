import csv
from app.schemas.schemas import PredictRequest, TrainRequest


TRAIN_COLUMNS = [
    "sex",
    "age",
    "side",
    "BW",
    "Ht",
    "BMI",
    "IKDC pre",
    "IKDC 3 m",
    "IKDC 6 m",
    "IKDC 1 Y",
    "IKDC 2 Y",
    "Lysholm pre",
    "Lysholm 3 m",
    "Lysholm 6 m",
    "Lysholm 1 Y",
    "Lysholm 2 Y",
    "Pre KL grade",
    "Post KL grade 2 Y",
    "MRI healing 1 Y",
    "MM extrusion pre",
    "MM extrusion post",
    "MM gap",
    "Degenerative meniscus",
    "medial femoral condyle",
    "medial tibial condyle",
    "lateral femoral condyle",
    "lateral tibial condyle",
]


def validate_predict_request(data: dict) -> bool:
    try:
        if data["sex"] not in [0, 1]:
            return False
        if not (0 <= data["age"] <= 120):
            return False
        if data["side"] not in [1, 2]:
            return False

        numeric_fields = [
            field
            for field, type_ in PredictRequest.__annotations__.items()
            if type_ in (int, float) and field not in ["sex", "age", "side"]
        ]

        for field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False
    except (ValueError, KeyError, TypeError) as e:
        return False

    return True


def validate_train_request_csv(file_path: str) -> bool:
    try:
        with open(file_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)

            if not set(TRAIN_COLUMNS).issubset(reader.fieldnames):
                return False

            for row in reader:
                converted_row = convert_csv_row_types(row)
                if not validate_train_request(converted_row):
                    return False

    except Exception as e:
        print(f"Exception: {e}")
        return False

    return True


def validate_train_request(data: dict) -> bool:
    try:
        if data["sex"] not in [0, 1]:
            return False
        if not (0 <= data["age"] <= 120):
            return False
        if data["side"] not in [1, 2]:
            return False

        numeric_fields = [
            field
            for field, type_ in TrainRequest.__annotations__.items()
            if type_ in (int, float) and field not in ["sex", "age", "side"]
        ]

        for field in numeric_fields:
            if not isinstance(data[field], (int, float)) or data[field] < 0:
                return False
    except (ValueError, KeyError, TypeError) as e:
        return False

    return True


def convert_csv_row_types(row: dict) -> dict:
    try:
        return {
            "sex": int(row["sex"]),
            "age": int(row["age"]),
            "side": int(row["side"]),
            "BW": float(row["BW"]),
            "Ht": float(row["Ht"]),
            "BMI": float(row["BMI"]),
            "IKDC_pre": float(row["IKDC pre"]),
            "IKDC_3_m": float(row["IKDC 3 m"]),
            "IKDC_6_m": float(row["IKDC 6 m"]),
            "IKDC_1_Y": float(row["IKDC 1 Y"]),
            "IKDC_2_Y": float(row["IKDC 2 Y"]),
            "Lysholm_pre": float(row["Lysholm pre"]),
            "Lysholm_3_m": float(row["Lysholm 3 m"]),
            "Lysholm_6_m": float(row["Lysholm 6 m"]),
            "Lysholm_1_Y": float(row["Lysholm 1 Y"]),
            "Lysholm_2_Y": float(row["Lysholm 2 Y"]),
            "Pre_KL_grade": float(row["Pre KL grade"]),
            "Post_KL_grade_2_Y": float(row["Post KL grade 2 Y"]),
            "MRI_healing_1_Y": float(row["MRI healing 1 Y"]),
            "MM_extrusion_pre": float(row["MM extrusion pre"]),
            "MM_extrusion_post": float(row["MM extrusion post"]),
            "MM_gap": float(row["MM gap"]),
            "Degenerative_meniscus": float(row["Degenerative meniscus"]),
            "medial_femoral_condyle": float(row["medial femoral condyle"]),
            "medial_tibial_condyle": float(row["medial tibial condyle"]),
            "lateral_femoral_condyle": float(row["lateral femoral condyle"]),
            "lateral_tibial_condyle": float(row["lateral tibial condyle"]),
        }

    except ValueError:
        return {}


def convert_csv_row_ten_types(row: dict) -> dict:
    try:
        return {
            "sex": int(row["sex"]),
            "age": int(row["age"]),
            "side": int(row["side"]),
            "BW": float(row["BW"]),
            "Ht": float(row["Ht"]),
            "BMI": float(row["BMI"]),
            "IKDC_pre": float(row["IKDC pre"]),
            "Lysholm_pre": float(row["Lysholm pre"]),
            "Pre_KL_grade": float(row["Pre KL grade"]),
            "MM_extrusion_pre": float(row["MM extrusion pre"]),
        }

    except ValueError:
        return {}
