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


def augment_smogn(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()

    added_df = smogn.smoter(
        ## main arguments
        data=df,
        y=target_column,
        k=10,  ## Increase neighbors
        pert=0.2,  ## Higher perturbation
        samp_method="extreme",  ## Focus on minority oversampling
        drop_na_col=True,
        drop_na_row=True,
        replace=True,  ## Allow re-sampling
        ## phi relevance arguments
        rel_thres=0.4,
        rel_method="manual",
        rel_ctrl_pts_rg=[
            [36, 1, 0],  ## Keep minority oversampling
            [50, 0.2, 0],  ## Reduce under-sampling
            [55, 0.2, 0],  ## Reduce under-sampling
            [70, 0.1, 0],  ## Reduce under-sampling
        ],
    )
    print(added_df.head(3))
    augmented_df = pd.concat([df, added_df], ignore_index=True)
    return augmented_df


def noise_augmentation(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    df = df.copy()
    X = df.drop(
        target_column, axis=1
    ).values  # Assuming target_column is your target column
    y = df[target_column].values

    n_synthetic = 50
    synthetic_data = []
    synthetic_labels = []
    feature_names = df.drop(target_column, axis=1).columns

    # 1. Gaussian Noise
    print("Generating samples with Gaussian noise...")
    for i in range(len(X)):
        for _ in range(3):  # Generate 3 noisy samples per original sample
            noise = np.random.normal(
                0, 0.05, size=X.shape[1]
            )  # Reduced noise level to 0.05
            synthetic_x = X[i] + noise
            # Ensure values stay within reasonable bounds
            synthetic_x = np.clip(synthetic_x, X.min(axis=0), X.max(axis=0))
            synthetic_data.append(synthetic_x)
            synthetic_labels.append(y[i])

    # 2. Interpolation between samples
    print("Generating interpolated samples...")
    for _ in range(n_synthetic):
        idx1, idx2 = np.random.randint(0, len(X), 2)
        alpha = np.random.random()
        synthetic_x = X[idx1] * alpha + X[idx2] * (1 - alpha)
        synthetic_y = y[idx1] * alpha + y[idx2] * (1 - alpha)
        synthetic_data.append(synthetic_x)
        synthetic_labels.append(synthetic_y)

    # 3. Slight scaling
    print("Generating scaled samples...")
    for i in range(len(X)):
        scale = np.random.uniform(0.95, 1.05, size=X.shape[1])  # Reduced scale range
        synthetic_x = X[i] * scale
        synthetic_data.append(synthetic_x)
        synthetic_labels.append(y[i])

    # Convert to numpy arrays
    synthetic_data = np.array(synthetic_data)
    synthetic_labels = np.array(synthetic_labels)

    # Create DataFrame with synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)
    synthetic_df[target_column] = synthetic_labels

    # Combine original and synthetic data
    final_df = pd.concat(
        [df, synthetic_df], ignore_index=True  # Original data  # Synthetic data
    )

    return final_df
