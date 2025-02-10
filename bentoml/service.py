import bentoml
import torch
from bentoml.io import NumpyNdarray
from regression_net import RegressionNet
import numpy as np

# Load your model
model = RegressionNet(input_dim=10, hidden_dim=177, num_layers=4, dropout=0.15)
model.load_state_dict(torch.load("../model_artifacts/regression_model.pth", map_location="cpu"))
model.eval()

# Save the model with BentoML
bento_model = bentoml.pytorch.save_model(
    name="regression_model",
    model=model,
    signatures={"__call__": {"batchable": False}},
)

# Create a service with a runner
svc = bentoml.Service("my_pytorch_service", runners=[bento_model.to_runner()])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def predict(input_arr: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(input_arr).float()
    with torch.no_grad():
        result = model(tensor)
    return result.numpy()
