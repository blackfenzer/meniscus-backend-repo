import bentoml
import torch
from bentoml.io import NumpyNdarray
from regression_net import RegressionNet
import numpy as np

# Load your model
# model = RegressionNet(input_dim=10, hidden_dim=177, num_layers=4, dropout=0.15)
# model.load_state_dict(
#     torch.load("../model_artifacts/regression_model.pth", map_location="cpu")
# )
# model.eval()

import torch
import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
from sklearn.preprocessing import StandardScaler

# Allowlist your custom model class so that torch.load() can deserialize it safely.
torch.serialization.add_safe_globals([RegressionNet])
torch.serialization.add_safe_globals([StandardScaler])


@bentoml.service
class RegressionService:
    """
    A BentoML service that loads a PyTorch model and its associated scaler
    from a checkpoint file, then serves predictions.
    """

    def __init__(self):
        # Load the model and scaler once when the service starts.
        self.model, self.scaler = self._load_model_and_scaler()

    def _load_model_and_scaler(self):
        """
        Loads the model state dict and scaler from a checkpoint file.
        The checkpoint is expected to be a dictionary containing:
          - 'model_state_dict': the PyTorch state dict of the model.
          - 'scaler': an instance of a fitted scaler (e.g., StandardScaler).
        """
        checkpoint_path = (
            "../model_artifacts/regression_model.pth"  # Ensure this path is correct.
        )
        checkpoint = torch.load(
            checkpoint_path, map_location=torch.device("cpu"), weights_only=False
        )

        # Initialize the model architecture with the same hyperparameters used during training.
        model = RegressionNet(input_dim=10, hidden_dim=151, num_layers=2, dropout=0.15)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set the model to evaluation mode.

        scaler = checkpoint["scaler"]
        return model, scaler

    @bentoml.api
    def predict(self, input_data) -> np.ndarray:
        """
        API endpoint to get predictions.
          - `input_data` is expected to be a NumPy array (e.g., a batch of samples).
          - The input is transformed using the previously fitted scaler.
          - The transformed input is converted to a torch tensor and passed through the model.
          - The model's output is returned as a NumPy array.
        """
        try:
            features = [
                input_data["sex"],
                input_data["age"],
                input_data["side"],
                input_data["BW"],
                input_data["Ht"],
                input_data["BMI"],
                input_data["IKDC pre"],
                input_data["Lysholm pre"],
                input_data["Pre KL grade"],
                input_data["MM extrusion pre"],
            ]
        except KeyError as e:
            return {"error": f"Missing field: {e.args[0]}"}
        features_array = np.array([features])
        
        
        # Apply the same scaling as during training.
        transformed_input = self.scaler.transform(features_array)
        tensor_input = torch.tensor(transformed_input, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(tensor_input)

        return {"prediction": output.detach().cpu().numpy().tolist()}
