import torch
import sys
from pathlib import Path
git_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET'#jacob path
sys.path.append(str(git_dir))

from UNET3D.unet_model.unet_model import UNet3D

model = UNet3D(n_channels=7, n_classes=4)

# Counting the number of parameters
total_params = sum(p.numel() for p in model.parameters())

# Counting the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
