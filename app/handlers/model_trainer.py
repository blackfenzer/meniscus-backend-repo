import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from app.handlers.augment_handler import (
    augment_smogn,
    missing_imputation,
    noise_augmentation,
)
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import numpy as np


TRAIN_COLUMNS = [
    "sex",
    "age",
    "side",
    "BW",
    "Ht",
    "BMI",
    "IKDC pre",
    "Lysholm pre",
    "Pre KL grade",
    "MM extrusion pre",
    "IKDC 2 Y",
]


class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(RegressionNet, self).__init__()
        layers = []

        # First layer: input_dim → hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim → 1
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def train_model_from_csv(csv_bytes: bytes):
    # Convert bytes to pandas DataFrame
    try:
        target_column = "IKDC 2 Y"
        csv_data = BytesIO(csv_bytes)
        df = pd.read_csv(csv_data)
        df = missing_imputation(df)
        df = augment_smogn(df, target_column)
        df = noise_augmentation(df, target_column)
        print("CSV Shape:", df.shape)  # Debugging log

        # Ensure only required columns are used
        df = df[TRAIN_COLUMNS]  # Drop other columns

        # Ensure CSV has the correct columns
        if df.shape[1] < len(TRAIN_COLUMNS):
            raise ValueError("CSV does not have the required number of columns")
        X = df.drop(target_column, axis=1).values  # Features
        y = df[target_column].values  # Target

        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        print("X_tensor shape:", X_tensor.shape)  # Debugging log
        print("y_tensor shape:", y_tensor.shape)  # Debugging log

        # Initialize model with dynamic input dimension
        input_dim = X.shape[1]  # Dynamically set input dimension
        model = RegressionNet(
            input_dim=input_dim, hidden_dim=151, num_layers=2, dropout=0.15
        )

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train for 10 epochs
        for _ in range(10):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
        return model, scaler

    except Exception as e:
        print(f"Error during training: {e}")
        # Optionally: re-raise or handle error as needed
        raise e


def train_model_with_kfold(csv_bytes: bytes, n_splits: int = 5):
    """
    Upgraded training function that uses k-fold cross validation to reduce overfitting.
    It prints R² and RMSE metrics for each fold.
    """
    try:
        target_column = "IKDC 2 Y"
        csv_data = BytesIO(csv_bytes)
        df = pd.read_csv(csv_data)
        df = missing_imputation(df)
        df = augment_smogn(df, target_column)
        df = noise_augmentation(df, target_column)
        print("CSV Shape:", df.shape)

        df = df[TRAIN_COLUMNS]
        if df.shape[1] < len(TRAIN_COLUMNS):
            raise ValueError("CSV does not have the required number of columns")

        X = df.drop(target_column, axis=1).values
        y = df[target_column].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        r2_scores, rmse_scores = [], []
        fold = 1

        for train_index, val_index in kf.split(X):
            print(f"\nFold {fold}")
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            # Convert fold data to tensors
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

            # Initialize a new model for each fold
            input_dim = X.shape[1]
            model = RegressionNet(
                input_dim=input_dim, hidden_dim=151, num_layers=2, dropout=0.15
            )
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Training loop for this fold
            epochs = 200
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                predictions = model(X_train_tensor)
                loss = criterion(predictions, y_train_tensor)
                loss.backward()
                optimizer.step()
                # Optional: Implement early stopping here if needed

            # Evaluate on validation fold
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor).numpy().flatten()

            fold_r2 = r2_score(y_val, val_predictions)
            fold_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            print(f"Fold {fold} - R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}")

            r2_scores.append(fold_r2)
            rmse_scores.append(fold_rmse)
            fold += 1

        print("\n=== Cross Validation Results ===")
        print(f"Average R²: {np.mean(r2_scores):.4f} (± {np.std(r2_scores):.4f})")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f} (± {np.std(rmse_scores):.4f})")

        # Optionally, retrain on the full dataset or return metrics for further analysis.
        return model, scaler, np.mean(rmse_scores), np.mean(r2_scores)

    except Exception as e:
        print(f"Error during k-fold training: {e}")
        raise e
