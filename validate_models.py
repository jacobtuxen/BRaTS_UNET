import nibabel as nib
import numpy as np
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from skimage.util import montage
from PIL import Image

from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics import JaccardIndex, ConfusionMatrix, Dice


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, dropout = 0.0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dropout = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True, dropout = 0.0):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2,x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False, scale_channels=1):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = (DoubleConv(n_channels, 32//scale_channels, dropout=0.1))
        self.down1 = (Down(32//scale_channels, 64//scale_channels, dropout=0.1))
        self.down2 = (Down(64//scale_channels, 128//scale_channels, dropout=0.2))
        self.down3 = (Down(128//scale_channels, 256//scale_channels, dropout=0.3))
        factor = 2 if trilinear else 1
        self.up1 = (Up(256//scale_channels, (128//scale_channels) // factor, trilinear, dropout=0.1))
        self.up2 = (Up(128//scale_channels, (64//scale_channels) // factor, trilinear, dropout=0.2))
        self.up3 = (Up(64//scale_channels, (32//scale_channels) // factor, trilinear, dropout=0.3))
        self.outc = (OutConv(32//scale_channels, n_classes))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path, binary='WT'): #(WT, TC, MT)
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.binary = binary
        self.extensions = ['flair.nii', 't1ce.nii', 't2.nii','seg.nii']

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data_paths = [self.data_dir / patient_id / f'{patient_id}_{data_id}' for data_id in self.extensions]
        data = [self.load_nifti_file(path) for path in data_paths]
        target = torch.from_numpy(np.where(data[-1]==4, 3, data[-1])).long()
        if self.binary == 'WT':
            target = torch.where(target==0, 0, 1)
        elif self.binary == 'MT':
            target = torch.where(target==3, 1, target)
            target = torch.where(target==2, 0, target)
        elif self.binary == 'TC':
            target = torch.where(target==1, 1, 0)
        else:
            raise ValueError('binary must be one of: WT, MT, TC')
        data = torch.cat([torch.from_numpy(data[i]).unsqueeze(0) for i in range(len(self.extensions)-1)], dim=0)

        start_idx = 56
        end_idx = 184
        start_idx_height = 13
        end_idx_height = 141
        
        data = data[:,start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]
        target = target[start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]

        #normalize data in each channel min max normalization
        for i in range(data.shape[0]):
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

        return data, target, patient_id


# 1. Load data
data_dir = Path(str('/work3/s211469/data'))
val_ids = np.loadtxt(data_dir / 'val_ids.txt', dtype=str)

# 2. Create dataset for validation
val_set_WT = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='WT')
val_set_TC = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='TC')
val_set_MT = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='MT')

models_path = Path(str('/work3/s211469/models'))

start_idx = 56
end_idx = 184
start_idx_height = 13
end_idx_height = 141

true_masks = [nib.load(data_dir / val_ids[val_idx] / f'{val_ids[val_idx]}_seg.nii').get_fdata() for val_idx in range(len(val_ids))]
true_masks = [torch.from_numpy(np.where(true_masks[val_idx]==4, 3, true_masks[val_idx])).long() for val_idx in range(len(val_ids))]
true_masks = [true_masks[val_idx][start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height] for val_idx in range(len(val_ids))]

# 3 Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_TC = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
model_WT = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
model_MT = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)

model_TC.load_state_dict(torch.load(models_path / 'model_TC_baseline.pt'))
model_WT.load_state_dict(torch.load(models_path / 'model_WT_baseline.pt'))
model_MT.load_state_dict(torch.load(models_path / 'model_MT_baseline.pt'))

model_TC.to(device=device)
model_WT.to(device=device)
model_MT.to(device=device)

models = [model_WT, model_TC, model_MT]
models_names = ['WT', 'TC', 'MT']
# Create dataset
val_sets = [val_set_WT, val_set_TC, val_set_MT]

predictions_TC = []
predictions_WT = []
predictions_MT = []

for val_idx in range(len(val_set_TC)):
    print(f'Predicting patient {val_idx+1} of {len(val_set_TC)}')
    image, target, patient_id = val_set_TC[val_idx]
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)
    model_TC.eval()
    with torch.no_grad():
        prediction = model_TC(image)
    prediction_int = np.squeeze(np.argmax(prediction.detach().cpu().numpy(), axis=1))
    predictions_TC.append(prediction_int)

for val_idx in range(len(val_set_WT)):
    print(f'Predicting patient {val_idx+1} of {len(val_set_WT)}')
    image, target, patient_id = val_set_WT[val_idx]
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)
    model_WT.eval()
    with torch.no_grad():
        prediction = model_WT(image)
    prediction_int = np.squeeze(np.argmax(prediction.detach().cpu().numpy(), axis=1))
    predictions_WT.append(prediction_int)

for val_idx in range(len(val_set_MT)):
    print(f'Predicting patient {val_idx+1} of {len(val_set_MT)}')
    image, target, patient_id = val_set_MT[val_idx]
    image = image.unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)
    model_MT.eval()
    with torch.no_grad():
        prediction = model_MT(image)
    prediction_int = np.squeeze(np.argmax(prediction.detach().cpu().numpy(), axis=1))
    predictions_MT.append(prediction_int)


total_preds = [torch.tensor(sum((predictions_WT[val_idx], predictions_TC[val_idx], predictions_MT[val_idx]))).unsqueeze(0) for val_idx in range(len(val_ids))]
jaccard = JaccardIndex(task = 'multiclass', num_classes=4).to(device=device)
dice = Dice(num_classes=4).to(device=device)

jaccard_scores = [jaccard(total_preds[val_idx].to(device=device), torch.tensor(true_masks[val_idx]).unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_ids))]
dice_scores = [dice(total_preds[val_idx].to(device=device), torch.tensor(true_masks[val_idx]).unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_ids))]

np.save(data_dir / 'total_jaccard_scores.npy', jaccard_scores)
np.save(data_dir / 'total_dice_scores.npy', dice_scores)



# jaccard_scores_TC = [jaccard(predictions_TC[val_idx].to(device=device), val_set_TC[val_idx][1].unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_set_TC))]
# jaccard_scores_WT = [jaccard(predictions_WT[val_idx].to(device=device), val_set_WT[val_idx][1].unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_set_WT))]
# jaccard_scores_MT = [jaccard(predictions_MT[val_idx].to(device=device), val_set_MT[val_idx][1].unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_set_MT))]

# dice_scores_TC = [dice(predictions_TC[val_idx].to(device=device), val_set_TC[val_idx][1].unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_set_TC))]
# dice_scores_WT = [dice(predictions_WT[val_idx].to(device=device), val_set_WT[val_idx][1].unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_set_WT))]
# dice_scores_MT = [dice(predictions_MT[val_idx].to(device=device), val_set_MT[val_idx][1].unsqueeze(0).to(device=device)).cpu().numpy() for val_idx in range(len(val_set_MT))]

# np.save(data_dir / 'jaccard_scores_TC.npy', jaccard_scores_TC)
# np.save(data_dir / 'jaccard_scores_WT.npy', jaccard_scores_WT)
# np.save(data_dir / 'jaccard_scores_MT.npy', jaccard_scores_MT)

# np.save(data_dir / 'dice_scores_TC.npy', dice_scores_TC)
# np.save(data_dir / 'dice_scores_WT.npy', dice_scores_WT)
# np.save(data_dir / 'dice_scores_MT.npy', dice_scores_MT)

# print('Jaccard scores TC: ', jaccard_scores_TC)
# print('Jaccard scores WT: ', jaccard_scores_WT)
# print('Jaccard scores MT: ', jaccard_scores_MT)

# print('Dice scores TC: ', dice_scores_TC)
# print('Dice scores WT: ', dice_scores_WT)
# print('Dice scores MT: ', dice_scores_MT)
