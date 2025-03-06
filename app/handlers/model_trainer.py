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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
import numpy as np
from app.handlers.augment_handler import augment_data
from torch.utils.data import Dataset, DataLoader
import smogn
import random

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
    "MM gap",
    "Degenerative meniscus",
    "medial femoral condyle",
    "medial tibial condyle",
    "lateral femoral condyle",
    "lateral tibial condyle",
]


class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(RegressionNet, self).__init__()
        layers = []

        # First layer: input_dim → hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        # layers.append(nn.Dropout(dropout))

        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(dropout))

        # Output layer: hidden_dim → 1
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# def train_model_from_csv(csv_bytes: bytes):
#     # Convert bytes to pandas DataFrame
#     try:
#         target_column = "IKDC 2 Y"
#         csv_data = BytesIO(csv_bytes)
#         df = pd.read_csv(csv_data)
#         df = missing_imputation(df)
#         df = augment_smogn(df, target_column)
#         df = noise_augmentation(df, target_column)
#         print("CSV Shape:", df.shape)  # Debugging log

#         # Ensure only required columns are used
#         df = df[TRAIN_COLUMNS]  # Drop other columns

#         # Ensure CSV has the correct columns
#         if df.shape[1] < len(TRAIN_COLUMNS):
#             raise ValueError("CSV does not have the required number of columns")
#         X = df.drop(target_column, axis=1).values  # Features
#         y = df[target_column].values  # Target

#         # Normalize features
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)

#         # Convert to PyTorch tensors
#         X_tensor = torch.tensor(X, dtype=torch.float32)
#         y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

#         print("X_tensor shape:", X_tensor.shape)  # Debugging log
#         print("y_tensor shape:", y_tensor.shape)  # Debugging log

#         # Initialize model with dynamic input dimension
#         input_dim = X.shape[1]  # Dynamically set input dimension
#         model = RegressionNet(
#             input_dim=input_dim, hidden_dim=151, num_layers=2, dropout=0.15
#         )

#         # Training setup
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)

#         # Train for 10 epochs
#         for _ in range(10):
#             optimizer.zero_grad()
#             predictions = model(X_tensor)
#             loss = criterion(predictions, y_tensor)
#             loss.backward()
#             optimizer.step()
#         return model, scaler

#     except Exception as e:
#         print(f"Error during training: {e}")
#         # Optionally: re-raise or handle error as needed
#         raise e


# def train_model_with_kfold(csv_bytes: bytes, n_splits: int = 5):
#     """
#     Upgraded training function that uses k-fold cross validation to reduce overfitting.
#     It prints R² and RMSE metrics for each fold.
#     """
#     try:
#         target_column = "IKDC 2 Y"
#         csv_data = BytesIO(csv_bytes)
#         df = pd.read_csv(csv_data)
#         df = missing_imputation(df)
#         df = augment_smogn(df, target_column)
#         df = noise_augmentation(df, target_column)
#         print("CSV Shape:", df.shape)

#         df = df[TRAIN_COLUMNS]
#         if df.shape[1] < len(TRAIN_COLUMNS):
#             raise ValueError("CSV does not have the required number of columns")

#         X = df.drop(target_column, axis=1).values
#         y = df[target_column].values

#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)

#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#         r2_scores, rmse_scores = [], []
#         fold = 1

#         for train_index, val_index in kf.split(X):
#             print(f"\nFold {fold}")
#             X_train, X_val = X[train_index], X[val_index]
#             y_train, y_val = y[train_index], y[val_index]

#             # Convert fold data to tensors
#             X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#             y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
#             X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
#             y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

#             # Initialize a new model for each fold
#             input_dim = X.shape[1]
#             model = RegressionNet(
#                 input_dim=input_dim, hidden_dim=151, num_layers=2, dropout=0.15
#             )
#             criterion = nn.MSELoss()
#             optimizer = optim.Adam(model.parameters(), lr=0.001)

#             # Training loop for this fold
#             epochs = 200
#             for epoch in range(epochs):
#                 model.train()
#                 optimizer.zero_grad()
#                 predictions = model(X_train_tensor)
#                 loss = criterion(predictions, y_train_tensor)
#                 loss.backward()
#                 optimizer.step()
#                 # Optional: Implement early stopping here if needed

#             # Evaluate on validation fold
#             model.eval()
#             with torch.no_grad():
#                 val_predictions = model(X_val_tensor).numpy().flatten()

#             fold_r2 = r2_score(y_val, val_predictions)
#             fold_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
#             print(f"Fold {fold} - R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}")

#             r2_scores.append(fold_r2)
#             rmse_scores.append(fold_rmse)
#             fold += 1

#         print("\n=== Cross Validation Results ===")
#         print(f"Average R²: {np.mean(r2_scores):.4f} (± {np.std(r2_scores):.4f})")
#         print(f"Average RMSE: {np.mean(rmse_scores):.4f} (± {np.std(rmse_scores):.4f})")

#         # Optionally, retrain on the full dataset or return metrics for further analysis.
#         return model, scaler, np.mean(rmse_scores), np.mean(r2_scores)


#     except Exception as e:
#         print(f"Error during k-fold training: {e}")
#         raise e
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Train Model Function
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    early_stopping_patience=10,
):
    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch).squeeze()
                running_val_loss += criterion(y_pred, y_batch).item()
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return train_losses, val_losses, best_val_loss


# Function to evaluate model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch).squeeze()
            test_loss += criterion(y_pred, y_batch).item()

            all_preds.extend(y_pred.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    test_loss /= len(test_loader)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)

    return (
        {"loss": test_loss, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
        all_preds,
        all_targets,
    )


def train_with_best_params(
    csv_bytes: bytes, best_params, device="cuda" if torch.cuda.is_available() else "cpu"
):
    SEED = 27
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_column = "IKDC 2 Y"
    drop_columns = [
        "Lysholm 2 Y",
        "Post KL grade 2 Y",
        "IKDC 3 m",
        "IKDC 6 m",
        "IKDC 1 Y",
        "Lysholm 3 m",
        "Lysholm 6 m",
        "Lysholm 1 Y",
        "MRI healing 1 Y",
        "MM extrusion post",
    ]

    # Create training DataFrame for augmentation
    try:
        csv_data = BytesIO(csv_bytes)
        df = pd.read_csv(csv_data)
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Split data into training and testing sets BEFORE augmentation or scaling
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("X_train shape:", X_train.shape)
        cols = [f"feature_{i}" for i in range(X_train.shape[1])]
        train_df = pd.DataFrame(X_train, columns=cols)
        train_df["target"] = y_train  # 'target' is temporary column name

        # Get box plot stats for the target variable
        arr = smogn.box_plot_stats(train_df["target"])["stats"]

        # Augment the training data using parameters from boxplot stats
        augmented_df = augment_data(train_df, "target", arr[0], arr[1], arr[2], arr[3])

        # Extract augmented features and target
        X_train_aug = augmented_df.drop(columns=["target"]).values
        y_train_aug = augmented_df["target"].values

        # Scale the augmented training data and test data
        scaler = StandardScaler()
        X_train_aug_scaled = scaler.fit_transform(X_train_aug)
        X_test_scaled = scaler.transform(X_test)

        # Create datasets and dataloaders
        train_dataset = TabularDataset(X_train_aug_scaled, y_train_aug)
        test_dataset = TabularDataset(X_test_scaled, y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=best_params.get("batch_size", 16), shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=best_params.get("batch_size", 16)
        )

        # Create model with best parameters
        input_dim = X_train.shape[1]
        model = RegressionNet(
            input_dim=input_dim,
            hidden_dim=best_params.get("hidden_dim", 64),
            num_layers=best_params.get("num_layers", 2),
        ).to(device)

        # Optimizer and loss function
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params.get("lr", 0.001),
            weight_decay=best_params.get("weight_decay", 0.0001),
        )
        criterion = nn.MSELoss()

        # Train model
        train_losses, val_losses, _ = train_model(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            device,
            epochs=100,
            early_stopping_patience=50,
        )

        # Evaluate model
        test_metrics, predictions, targets = evaluate_model(
            model, test_loader, criterion, device
        )
        new_sample = np.array(
            [[0, 62, 1, 74.5, 165.0, 27.36, 56, 80, 2, 5.20, 3.55, 1, 4, 0, 0, 0]]
        )
        # Scale the new sample using the same scaler from training
        new_sample_scaled = scaler.transform(new_sample)
        new_sample_tensor = torch.tensor(new_sample_scaled, dtype=torch.float32)
        print("New sample tensor shape:", new_sample_scaled)
        # Make prediction with the trained model
        model.eval()
        with torch.no_grad():
            new_prediction = model(new_sample_tensor).squeeze()

        print("New prediction on the sample:", new_prediction.cpu().numpy())

        return (
            model,
            train_losses,
            val_losses,
            test_metrics,
            predictions,
            targets,
            scaler,
        )
    except Exception as e:
        print(f"Error during k-fold training: {e}")
        raise e


# def train_model_with_kfold(
#     csv_bytes: bytes,
#     best_params,
#     n_splits: int = 5,
#     device="cuda" if torch.cuda.is_available() else "cpu",
# ):
#     """
#     Upgraded training function that uses k-fold cross-validation with best parameters.
#     It prints R² and RMSE metrics for each fold.
#     """
#     # Set random seed
#     SEED = 27
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     torch.cuda.manual_seed(SEED)

#     print(f"Using device: {device}")
#     try:
#         target_column = "IKDC 2 Y"
#         csv_data = BytesIO(csv_bytes)
#         df = pd.read_csv(csv_data)
#         df = missing_imputation(df)
#         if df.isnull().sum().sum() > 0:
#             print(
#                 "Warning: Missing values detected after augmentation. Imputing again."
#             )
#             df = missing_imputation(df)
#         print("CSV Shape:", df.shape)

#         df = df[TRAIN_COLUMNS]
#         if df.shape[1] < len(TRAIN_COLUMNS):
#             raise ValueError("CSV does not have the required number of columns")

#         X = df.drop(target_column, axis=1).values
#         y = df[target_column].values

#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)

#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#         r2_scores, rmse_scores = [], []
#         fold = 1

#         for train_index, val_index in kf.split(X):
#             print(f"\nFold {fold}")
#             X_train, X_val = X[train_index], X[val_index]
#             y_train, y_val = y[train_index], y[val_index]

#             # Data augmentation
#             cols = [f"feature_{i}" for i in range(X_train.shape[1])]
#             train_df = pd.DataFrame(X_train, columns=cols)
#             train_df["target"] = y_train
#             arr = smogn.box_plot_stats(train_df["target"])["stats"]
#             augmented_df = augment_data(
#                 train_df, "target", arr[0], arr[1], arr[2], arr[3]
#             )

#             X_train_aug = augmented_df.drop(columns=["target"]).values
#             y_train_aug = augmented_df["target"].values

#             # Scale the augmented data
#             X_train_aug_scaled = scaler.fit_transform(X_train_aug)
#             X_val_scaled = scaler.transform(X_val)

#             # Create datasets and dataloaders
#             train_dataset = TabularDataset(X_train_aug_scaled, y_train_aug)
#             val_dataset = TabularDataset(X_val_scaled, y_val)
#             train_loader = DataLoader(
#                 train_dataset,
#                 batch_size=best_params.get("batch_size", 16),
#                 shuffle=True,
#             )
#             val_loader = DataLoader(
#                 val_dataset, batch_size=best_params.get("batch_size", 16)
#             )

#             # Initialize model
#             input_dim = X.shape[1]
#             model = RegressionNet(
#                 input_dim=input_dim,
#                 hidden_dim=best_params.get("hidden_dim", 64),
#                 num_layers=best_params.get("num_layers", 2),
#                 dropout=best_params.get("dropout", 0.2),
#             ).to(device)

#             optimizer = torch.optim.Adam(
#                 model.parameters(),
#                 lr=best_params.get("lr", 0.001),
#                 weight_decay=best_params.get("weight_decay", 0.0001),
#             )
#             criterion = nn.MSELoss()

#             # Train model
#             # train_losses, val_losses, _ = train_model(
#             #     model, train_loader, val_loader, optimizer, criterion, device,
#             #     epochs=100, early_stopping_patience=50
#             # )

#             # Evaluate model
#             val_metrics, predictions, targets = evaluate_model(
#                 model, val_loader, criterion, device
#             )
#             fold_r2 = val_metrics["r2"]
#             fold_rmse = val_metrics["rmse"]
#             print(f"Fold {fold} - R²: {fold_r2:.4f}, RMSE: {fold_rmse:.4f}")

#             r2_scores.append(fold_r2)
#             rmse_scores.append(fold_rmse)
#             fold += 1

#         print("\n=== Cross Validation Results ===")
#         print(f"Average R²: {np.mean(r2_scores):.4f} (± {np.std(r2_scores):.4f})")
#         print(f"Average RMSE: {np.mean(rmse_scores):.4f} (± {np.std(rmse_scores):.4f})")

#         return model, scaler, np.mean(rmse_scores), np.mean(r2_scores)

#     except Exception as e:
#         print(f"Error during k-fold training: {e}")
#         raise e


# def test_train_model(
#     csv_bytes: bytes,
#     best_params,
#     n_splits: int = 10,
# ):
#     SEED = 27
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(SEED)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     target_column = "IKDC 2 Y"
#     drop_columns = [
#         "Lysholm 2 Y",
#         "Post KL grade 2 Y",
#         "IKDC 3 m",
#         "IKDC 6 m",
#         "IKDC 1 Y",
#         "Lysholm 3 m",
#         "Lysholm 6 m",
#         "Lysholm 1 Y",
#         "MRI healing 1 Y",
#         "MM extrusion post",
#     ]

#     try:
#         # Load CSV and preprocess
#         csv_data = BytesIO(csv_bytes)
#         df = pd.read_csv(csv_data)
#         df = df.drop(columns=drop_columns)
#         X = df.drop(columns=[target_column]).values
#         y = df[target_column].values

#         # Split data into training and testing sets BEFORE augmentation or scaling
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         input_dim = X_train.shape[1]

#         # Prepare test loader (do not augment test set)
#         test_dataset = TabularDataset(X_test, y_test)
#         test_loader = DataLoader(
#             test_dataset, batch_size=best_params.get("batch_size", 16)
#         )

#         kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#         cv_scores = []

#         # For tracking the best fold across CV
#         global_best_val_loss = float("inf")
#         best_model_state = None
#         best_scaler = None
#         best_train_losses = None
#         best_val_losses = None

#         # Iterate through CV folds
#         for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
#             print(f"Training fold {fold+1}/{n_splits}")

#             # Split data for this fold
#             X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
#             y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

#             # Create DataFrame for augmentation (using temporary feature names)
#             cols = [f"feature_{i}" for i in range(X_train_fold.shape[1])]
#             fold_train_df = pd.DataFrame(X_train_fold, columns=cols)
#             fold_train_df["target"] = y_train_fold  # temporary target column

#             # Get box plot stats for the target variable
#             stats = smogn.box_plot_stats(fold_train_df["target"])["stats"]

#             # Augment only the training fold data
#             augmented_df = augment_data(
#                 fold_train_df, "target", stats[0], stats[1], stats[2], stats[3]
#             )
#             X_train_aug = augmented_df.drop(columns=["target"]).values
#             y_train_aug = augmented_df["target"].values

#             # Fit scaler on augmented training data and transform both training and validation data
#             scaler = StandardScaler()
#             X_train_aug_scaled = scaler.fit_transform(X_train_aug)
#             X_val_fold_scaled = scaler.transform(X_val_fold)

#             # Create datasets and loaders
#             train_dataset = TabularDataset(X_train_aug_scaled, y_train_aug)
#             val_dataset = TabularDataset(X_val_fold_scaled, y_val_fold)
#             train_loader = DataLoader(
#                 train_dataset,
#                 batch_size=best_params.get("batch_size", 16),
#                 shuffle=True,
#             )
#             val_loader = DataLoader(
#                 val_dataset, batch_size=best_params.get("batch_size", 16)
#             )

#             # Create model (RegressionNet should be defined elsewhere)
#             model = RegressionNet(
#                 input_dim,
#                 best_params.get("hidden_dim", 128),
#                 best_params.get("num_layers", 4),
#                 0,  # assuming dropout=0 here; adjust if needed
#             ).to(device)

#             # Set up optimizer and loss function
#             optimizer = torch.optim.Adam(
#                 model.parameters(),
#                 lr=best_params.get("lr", 0.01),
#                 weight_decay=best_params.get("weight_decay", 0.001),
#             )
#             criterion = nn.MSELoss()

#             # Train model on this fold
#             train_losses, val_losses, val_loss = train_model(
#                 model,
#                 train_loader,
#                 val_loader,
#                 optimizer,
#                 criterion,
#                 device,
#                 epochs=35,
#                 early_stopping_patience=20,
#             )
#             cv_scores.append(val_loss)
#             print(f"Fold {fold+1} validation loss: {val_loss:.6f}")

#             # Update global best if current fold is better
#             if val_loss < global_best_val_loss:
#                 global_best_val_loss = val_loss
#                 best_model_state = model.state_dict().copy()
#                 best_scaler = scaler
#                 best_train_losses = train_losses
#                 best_val_losses = val_losses

#         mean_val_loss = np.mean(cv_scores)
#         print(f"CV completed. Mean validation loss: {mean_val_loss:.6f}")

#         # Load best model state (from best fold) into a new model instance
#         best_model = RegressionNet(
#             input_dim,
#             best_params.get("hidden_dim", 128),
#             best_params.get("num_layers", 4),
#             0,
#         ).to(device)
#         best_model.load_state_dict(best_model_state)

#         # Evaluate the best model on the test set
#         test_metrics, predictions, targets = evaluate_model(
#             best_model, test_loader, criterion, device
#         )
#         print("Test metrics:", test_metrics)
#         return (
#             best_model,
#             best_train_losses,
#             best_val_losses,
#             test_metrics,
#             predictions,
#             targets,
#             best_scaler,
#         )

#     except Exception as e:
#         print(f"Error during k-fold training: {e}")
#         raise e
