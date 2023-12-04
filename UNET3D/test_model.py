import sys
sys.path.append('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET')
from data_loader import BrainDataset
from UNET3D.unet_model.unet_model import UNet3D
from pathlib import Path
import torch
from torch.utils.data import DataLoader
data_dir = Path(str('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET/data/archive'))
#is dir
patient_ids = ['BraTS2021_00495','BraTS2021_00495','BraTS2021_00495','BraTS2021_00495']
training_ids = patient_ids
wavelet = 'haar'
data_type = 'WT'#[WT,MT,TC]
batch_size = 1
train_set = BrainDataset(patient_ids=training_ids, data_dir=data_dir, binary=data_type)
loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
model = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
device = 'cpu'
model.to(device)

# Generate batch of data
for batch in train_loader:
    image, mask_true, patient_id = batch[0], batch[1], batch[2]
    image = image.to(device=device, dtype=torch.float32)
    mask_true = mask_true.to(device=device, dtype=torch.long)
    mask_pred = model(image)
    print(f'Mask true shape: {mask_true.shape}')
    print(f'Mask pred shape: {mask_pred.shape}')
    break