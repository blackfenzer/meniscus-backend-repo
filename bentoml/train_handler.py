from io import BytesIO
import joblib
from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import random
import smogn
import xgboost as xgb

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_COLUMN = "IKDC 2 Y"
DROP_COLUMNS = [
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
    "BMI",
    "lateral tibial condyle",
]


##################################
# Dataset and Model Definitions  #
##################################
class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class RegressionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        layers = []
        # First layer: Linear -> BatchNorm -> ReLU -> Dropout
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        # Hidden layers with BN, ReLU, Dropout
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        # Final layer: Output regression value
        layers.append(nn.Linear(hidden_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


##################################
# Data Augmentation Functions    #
##################################
def augment_smogn(
    df: pd.DataFrame, target_column: str, a=36, b=50, c=55, d=70
) -> pd.DataFrame:
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


def noise_augmentation(
    df: pd.DataFrame,
    target_column: str,
    categorical_cols: list = None,
    gaussian_prob: float = 0.7,
    gaussian_scale: float = 0.1,
    uniform_prob: float = 0.7,
    uniform_scale: float = 0.05,
    n_interp: int = None,
    scaling_prob: float = 0.7,
    scaling_range: tuple = (0.96, 1.06),
    target_noise_prob: float = 0.7,
    target_noise_scale: float = 0.1,
    random_seed: int = None,
) -> pd.DataFrame:
    if random_seed is not None:
        np.random.seed(random_seed)

    df_copy = df.copy()
    X = df_copy.drop(target_column, axis=1)
    y = df_copy[target_column]

    if categorical_cols is None:
        categorical_cols = []
    numerical_cols = [col for col in X.columns if col not in categorical_cols]

    feature_sds = X[numerical_cols].std()
    min_vals = X[numerical_cols].min()
    max_vals = X[numerical_cols].max()

    if n_interp is None:
        n_interp = len(X)

    synthetic_data = []
    synthetic_labels = []

    # Gaussian noise augmentation
    for i in range(len(X)):
        if np.random.rand() < gaussian_prob:
            noise = (
                np.random.normal(0, gaussian_scale, size=len(numerical_cols))
                * feature_sds
            )
            synthetic_x = X[numerical_cols].iloc[i].values + noise
            synthetic_x = np.clip(synthetic_x, min_vals, max_vals)
            synthetic_row = X.iloc[i].copy()
            synthetic_row[numerical_cols] = synthetic_x
            synthetic_data.append(synthetic_row)
            synthetic_labels.append(y.iloc[i])

    # Uniform noise augmentation
    for i in range(len(X)):
        if np.random.rand() < uniform_prob:
            noise = np.random.uniform(
                -uniform_scale, uniform_scale, size=len(numerical_cols)
            ) * (max_vals - min_vals)
            synthetic_x = X[numerical_cols].iloc[i].values + noise
            synthetic_x = np.clip(synthetic_x, min_vals, max_vals)
            synthetic_row = X.iloc[i].copy()
            synthetic_row[numerical_cols] = synthetic_x
            synthetic_data.append(synthetic_row)
            synthetic_labels.append(y.iloc[i])

    # Interpolation between random samples
    for _ in range(n_interp):
        idx1, idx2 = np.random.randint(0, len(X), 2)
        alpha = np.random.random()
        synthetic_x = X[numerical_cols].iloc[idx1] * alpha + X[numerical_cols].iloc[
            idx2
        ] * (1 - alpha)
        synthetic_y = y.iloc[idx1] * alpha + y.iloc[idx2] * (1 - alpha)
        synthetic_row = X.iloc[idx1].copy()
        synthetic_row[numerical_cols] = synthetic_x
        synthetic_data.append(synthetic_row)
        synthetic_labels.append(synthetic_y)

    # Scaling augmentation
    for i in range(len(X)):
        if np.random.rand() < scaling_prob:
            scale = np.random.uniform(
                scaling_range[0], scaling_range[1], size=len(numerical_cols)
            )
            synthetic_x = X[numerical_cols].iloc[i].values * scale
            synthetic_x = np.clip(synthetic_x, min_vals, max_vals)
            synthetic_row = X.iloc[i].copy()
            synthetic_row[numerical_cols] = synthetic_x
            synthetic_data.append(synthetic_row)
            synthetic_labels.append(y.iloc[i])

    # Target noise augmentation
    for i in range(len(X)):
        if np.random.rand() < target_noise_prob:
            noise = np.random.normal(0, target_noise_scale * y.std())
            synthetic_data.append(X.iloc[i])
            synthetic_labels.append(y.iloc[i] + noise)

    synthetic_df = pd.DataFrame(synthetic_data, columns=X.columns)
    synthetic_df[target_column] = synthetic_labels

    final_df = pd.concat([df_copy, synthetic_df], ignore_index=True)
    return final_df


def augment_data(df, target_column, a=36, b=50, c=55, d=70):
    smogn_df = augment_smogn(df, target_column, a, b, c, d)
    print("Shape after SMOGN:", smogn_df.shape)
    final_df = noise_augmentation(smogn_df, target_column)
    print("Shape after noise augmentation:", final_df.shape)
    return final_df


##################################
# Training & Evaluation Functions#
##################################
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
    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                running_val_loss += loss.item()
        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}"
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    avg_loss = total_loss / len(data_loader)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return (
        {"loss": avg_loss, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
        all_preds,
        all_targets,
    )


##################################
# Stratified K-Fold Cross Validation for Regression
##################################
def run_stratified_kfold(
    X,
    y,
    n_splits=5,
    augment=True,
    epochs=100,
    batch_size=32,
    model_params=None,
    optimizer_params=None,
):
    # For stratification in regression, we bin the target values
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n--- Fold {fold+1} ---")
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create a DataFrame from the training fold for augmentation
        cols = [f"feature_{i}" for i in range(X.shape[1])]
        train_df = pd.DataFrame(X_train_fold, columns=cols)
        train_df["target"] = y_train_fold

        if augment:
            augmented_df = augment_data(train_df, "target")
            X_train_aug = augmented_df.drop(columns=["target"]).values
            y_train_aug = augmented_df["target"].values
        else:
            X_train_aug, y_train_aug = X_train_fold, y_train_fold

        # Scale features (fit on training fold only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug)
        X_val_scaled = scaler.transform(X_val_fold)

        # Create datasets and loaders
        train_dataset = TabularDataset(X_train_scaled, y_train_aug)
        val_dataset = TabularDataset(X_val_scaled, y_val_fold)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Instantiate model
        input_dim = X.shape[1]
        hidden_dim = model_params.get("hidden_dim", 128)
        num_layers = model_params.get("num_layers", 4)
        dropout = model_params.get("dropout", 0.2)
        model = RegressionNet(input_dim, hidden_dim, num_layers, dropout).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_params.get("lr", 0.001),
            weight_decay=optimizer_params.get("weight_decay", 1e-5),
        )
        criterion = nn.MSELoss()

        # Train model on this fold
        model, train_losses, val_losses = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            device,
            epochs,
            early_stopping_patience=10,
        )

        metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Fold {fold+1} Metrics: {metrics}")
        fold_metrics.append(metrics)

    # Average metrics across folds
    avg_metrics = {
        key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]
    }
    print("\nAverage CV Metrics:")
    print(avg_metrics)
    return avg_metrics


##################################
# Main Pipeline                  #
##################################
def train_pipeline_regression(
    csv_bytes: bytes,
    model_params: dict = None,
    optimizer_params: dict = None,
    batch_size: int = 32,
    epochs: int = 100,
):
    SEED = 9
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    TARGET_COLUMN = "IKDC 2 Y"
    DROP_COLUMNS = [
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
        "BMI",
        "lateral tibial condyle",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    model_params = model_params or {"hidden_dim": 251, "num_layers": 8, "dropout": 0.1}
    optimizer_params = optimizer_params or {"lr": 0.01905, "weight_decay": 0.01754}

    # Load data (update data_path as needed)
    csv_data = BytesIO(csv_bytes)
    df = pd.read_csv(csv_data)
    df = df.drop(columns=DROP_COLUMNS)

    # Split into features and target
    feature_columns = df.drop(columns=[TARGET_COLUMN]).columns
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values

    # Split into training and test sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    # Run stratified k-fold cross validation on training set
    print("\nStarting Stratified K-Fold CV on Training Data")
    cv_metrics = run_stratified_kfold(
        X_train,
        y_train,
        n_splits=4,
        augment=True,
        epochs=epochs,
        batch_size=batch_size,
        model_params=model_params,
        optimizer_params=optimizer_params,
    )

    # After CV, retrain final model on full training set and evaluate on test set
    # Augment full training set
    cols = [f"feature_{i}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=cols)
    train_df["target"] = y_train
    augmented_df = augment_data(train_df, "target")
    X_train_aug = augmented_df.drop(columns=["target"]).values
    y_train_aug = augmented_df["target"].values

    # Scale features (fit on full training set)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_aug)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TabularDataset(X_train_scaled, y_train_aug)
    test_dataset = TabularDataset(X_test_scaled, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Instantiate final model
    input_dim = X_train.shape[1]
    final_model = RegressionNet(
        input_dim,
        model_params["hidden_dim"],
        model_params["num_layers"],
        model_params["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=optimizer_params["lr"],
        weight_decay=optimizer_params["weight_decay"],
    )
    criterion = nn.MSELoss()

    print("\nTraining final model on full training set...")
    final_model, train_losses, val_losses = train_model(
        final_model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device,
        epochs,
        early_stopping_patience=15,
    )
    final_metrics, predictions, targets = evaluate_model(
        final_model, test_loader, criterion, device
    )
    return (
        final_model,
        train_losses,
        val_losses,
        final_metrics,
        predictions,
        targets,
        scaler,
        input_dim,
    )

##################################
# Train using XG boost           #
##################################

def train_xg_boost(
    csv_bytes: bytes,
    param_grid: dict = None,
    batch_size: int = 32,
    test_size: float = 0.2,
    seed: int = 9,
):
    np.random.seed(seed)
    
    TARGET_COLUMN = "IKDC 2 Y"
    DROP_COLUMNS = [
        "Lysholm 2 Y", "Post KL grade 2 Y", "IKDC 3 m", "IKDC 6 m",
        "IKDC 1 Y", "Lysholm 3 m", "Lysholm 6 m", "Lysholm 1 Y",
        "MRI healing 1 Y", "MM extrusion post", "BMI", "lateral tibial condyle"
    ]
    
    # Load CSV data
    csv_data = BytesIO(csv_bytes)
    df = pd.read_csv(csv_data)
    df = df.drop(columns=DROP_COLUMNS).dropna()
    
    # Split features and target
    X = df.drop(columns=[TARGET_COLUMN]).values
    y = df[TARGET_COLUMN].values
    input_dim = X.shape[1]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define default hyperparameter grid if none is provided
    param_grid = {
        'n_estimators': [200], #[50, 200],
        'max_depth': [3], #[3, 7, 9],
        'learning_rate': [0.01], #[0.01, 0.001],
        'subsample': [0.5],
        'colsample_bytree': [1.0], #[0.9, 1.0],
        'gamma': [1],
        'min_child_weight': [5], #[5, 7]
    }
    
    # Define the XGBoost regressor
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=seed)
    
    # Setup KFold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_reg,
        param_grid=param_grid,
        scoring='neg_mean_absolute_error',
        cv=kf,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    
    # Re-fit best_model with an evaluation set to capture loss history.
    # We use MAE as the evaluation metric (you can choose another if you prefer)
    best_model.fit(
        X_train_scaled, y_train
    )
    
    # Predict on test data
    y_pred = best_model.predict(X_test_scaled)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    final_metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    
    final_model = best_model
    predictions = y_pred
    targets = y_test
    
    logger.info("return")
    return (
        final_model,
        0,
        0,
        final_metrics,
        predictions,
        targets,
        scaler,
        input_dim,
    )