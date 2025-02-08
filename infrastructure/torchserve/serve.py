# infrastructure/torchserve/serve.py
import os
import subprocess
from app.config import MODEL_DIR


def package_model_for_torchserve(
    model_version: int, model_name: str = "regression_net"
):
    model_file = os.path.join(MODEL_DIR, f"model_v{model_version}.pt")
    mar_file = f"{model_name}.mar"
    # Example command for torch-model-archiver
    cmd = [
        "torch-model-archiver",
        "--model-name",
        model_name,
        "--version",
        str(model_version),
        "--serialized-file",
        model_file,
        "--handler",
        "handler.py",  # Custom handler (you need to create one)
        "--export-path",
        "./model_store",
        "--extra-files",
        f"{MODEL_DIR}/scaler.pt",  # if you have a scaler or other dependencies
    ]
    subprocess.run(cmd, check=True)
    print(f"Model packaged as {mar_file} and saved to './model_store'")
