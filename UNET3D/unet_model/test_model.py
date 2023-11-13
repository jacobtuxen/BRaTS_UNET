import torch
import sys
from pathlib import Path
git_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET'#jacob path
sys.path.append(str(git_dir))
from UNET3D.unet_model.unet_model import UNet3D

images = torch.rand(1,4,160,160,160)
labels = torch.rand(1,160,160,160)

model = UNet3D(n_channels=4, n_classes=4)
output = model(images)
print(output.shape)