import torch
from safetensors.torch import load_file

path = "Qwen3.5-with-Spatial-Enhancer-Adapter/spatial_enhancer_adapter.pth"
state_dict = load_file(path)
enhancer_weights = {k: v for k, v in state_dict.items() if 'spatial_enhancer' in k}
# Save the extracted weights to a new file
torch.save(enhancer_weights, "spatial_enhancer_final.pth")
print("spatial_enhancer_final.pth saved successfully.")