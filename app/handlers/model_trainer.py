import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from io import BytesIO

TRAIN_COLUMNS = [
    "sex", "age", "side", "BW", "Ht", "BMI", "IKDC pre", "Lysholm pre", "Pre KL grade", "MM extrusion pre", "IKDC 2 Y"
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
    csv_data = BytesIO(csv_bytes)
    df = pd.read_csv(csv_data)

    print("CSV Shape:", df.shape)  # Debugging log

    # Ensure only required columns are used
    df = df[TRAIN_COLUMNS]  # Drop other columns

    # Ensure CSV has the correct columns
    if df.shape[1] < len(TRAIN_COLUMNS):
        raise ValueError("CSV does not have the required number of columns")
    X = df.drop("IKDC 2 Y", axis=1).values  # Features
    y = df["IKDC 2 Y"].values  # Target

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
    model = RegressionNet(input_dim=input_dim, hidden_dim=151, num_layers=2, dropout=0.15)

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
