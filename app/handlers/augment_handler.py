import pandas as pd
import smogn
import numpy as np

ALL_COLUMNS = [
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

CATEGORICAL_COLUMNS = ["sex", "side"]

NUMERIC_COLUMNS = [
    "age",
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

def missing_imputation(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
        else:
            print(f"Warning: column '{col}' not found in DataFrame")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            print(f"Warning: column '{col}' not found in DataFrame")

    return df


def augment_smogn(df: pd.DataFrame, target_column: str, a=36, b=50, c=55, d=70) -> pd.DataFrame:
    df = df.copy()
    added_df = smogn.smoter(
        data=df,
        y=target_column,
        k=10,
        pert=0.1,
        samp_method="extreme",
        drop_na_col=True,
        drop_na_row=True,
        replace=True,
        rel_thres=0.4,
        rel_method="manual",
        rel_ctrl_pts_rg=[
            [a, 1.5, 0],
            [b, 0.2, 0],
            [c, 0.2, 0],
            [d, 0.1, 0],
        ],
    )
    augmented_df = pd.DataFrame(added_df)
    return augmented_df

def noise_augmentation(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    feature_names = X.columns
    feature_sds = X.std()
    min_vals = X.min()
    max_vals = X.max()
    synthetic_data = []
    synthetic_labels = []
    n_synthetic = 50
    
    # 1. Gaussian Noise with SD scaling
    for i in range(len(X)):
        if np.random.rand() > 0.60:  # 30% chance to augment
            noise = np.random.normal(0, 0.1, size=len(feature_sds)) * feature_sds
            synthetic_x = X.iloc[i] + noise
            synthetic_x = np.clip(synthetic_x, min_vals, max_vals)
            synthetic_data.append(synthetic_x)
            synthetic_labels.append(y.iloc[i])
    
    # 2. Interpolation between samples
    for _ in range(n_synthetic):
        idx1, idx2 = np.random.randint(0, len(X), 2)
        alpha = np.random.random()
        synthetic_x = X.iloc[idx1] * alpha + X.iloc[idx2] * (1 - alpha)
        synthetic_y = y.iloc[idx1] * alpha + y.iloc[idx2] * (1 - alpha)
        synthetic_data.append(synthetic_x)
        synthetic_labels.append(synthetic_y)
    
    # 3. Slight scaling
    for i in range(len(X)):
        scale = np.random.uniform(0.97, 1.03, size=len(feature_names))
        synthetic_x = X.iloc[i] * scale
        synthetic_data.append(synthetic_x)
        synthetic_labels.append(y.iloc[i])
    
    # Create DataFrame with synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
    synthetic_df[target_column] = synthetic_labels
    
    # Combine original and synthetic data
    final_df = pd.concat([df, synthetic_df], ignore_index=True)
    return final_df

def augment_data(df, target_column, a=36, b=50, c=55, d=70):
    # First apply SMOGN augmentation
    smogn_df = augment_smogn(df, target_column, a, b, c, d)
    
    # Then apply noise augmentation
    final_df = noise_augmentation(smogn_df, target_column)
    
    return final_df