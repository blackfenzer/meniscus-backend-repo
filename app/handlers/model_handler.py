# import torch
# import torch.nn.functional as F
# from ts.torch_handler.base_handler import BaseHandler
# from regression_net import RegressionNet

# import sys
# import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# class RegressionHandler(BaseHandler):
#     def initialize(self, context):
#         self.manifest = context.manifest
#         self.model_dir = context.system_properties.get("model_dir")

#         model_path = f"{self.model_dir}/regression_model.pth"
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Load model
#         self.model = RegressionNet(
#             input_dim=10, hidden_dim=177, num_layers=4, dropout=0.15
#         )
#         self.model.load_state_dict(torch.load(model_path, map_location=self.device))
#         self.model.to(self.device)
#         self.model.eval()

#     def preprocess(self, data):
#         """Convert input JSON to tensor"""
#         input_data = torch.tensor(data, dtype=torch.float32).to(self.device)
#         return input_data

#     def inference(self, inputs):
#         """Run inference on input data"""
#         with torch.no_grad():
#             output = self.model(inputs)
#         return output.tolist()

#     def postprocess(self, outputs):
#         """Convert output tensor to JSON"""
#         return [{"prediction": pred} for pred in outputs]
