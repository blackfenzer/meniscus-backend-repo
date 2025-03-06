import torch.nn as nn


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
