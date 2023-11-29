import nibabel as nib
from pathlib import Path
import torch
import numpy as np
from torch.nn.functional import one_hot
from skimage.util import montage
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, random_split
""" Parts of the U-Net model """

import torch.nn as nn
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F

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
        data = data - data.min() / (1e-5 + (data.max() - data.min())) 

        return data, target, patient_id



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

def predictions_plot(image, mask_true, mask_pred, patient_id='BraTS2021_00495'):
    #Args:
    #mask_true: true binary mask is a tensor of size (1, 2, 128,128,128)
    #mask_pred: predicted binary mask is a tensor of size (1, 2, 128,128,128)
    #image: is a flair data tensor of size (128,128,128)
    #patient_id: is a string of the patient id
    assert len(mask_true.shape) == 5
    assert len(mask_pred.shape) == 5
    assert len(image.shape) == 3

    img_tensor = image.cpu().detach().numpy()
    mask_true_tensor = mask_true.squeeze()[1].squeeze().cpu().detach().numpy()
    mask_pred_tensor = mask_pred.squeeze()[1].squeeze().cpu().detach().numpy()

    #Pad zeros to size 160x160x160
    img_tensor = np.pad(img_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    mask_true_tensor = np.pad(mask_true_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    mask_pred_tensor = np.pad(mask_pred_tensor, ((16,16),(16,16),(16,16)), 'constant', constant_values=0)
    

    image = np.rot90(montage(img_tensor))
    mask_true = np.rot90(montage(mask_true_tensor))
    mask_pred = np.rot90(montage(mask_pred_tensor))


    intersection = np.logical_and(mask_true, mask_pred)
    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    plt.subplot(1,1,1)
    ax.imshow(image, cmap ='bone')
    ax.imshow(np.ma.masked_where(mask_true == False, mask_true),
                cmap='cool', alpha=0.9)
    ax.imshow(np.ma.masked_where(mask_pred == False, mask_pred),
                cmap='spring', alpha=0.6)
    ax.imshow(np.ma.masked_where(intersection == False, intersection),
                cmap='winter', alpha=1)
    plt.suptitle(f'Error plot patient: {patient_id}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(handles=[Line2D([0], [0], color='pink', lw=4, label='True'),
                       Line2D([0], [0], color='yellow', lw=4, label='Predicted'),
                       Line2D([0], [0], color='blue', lw=4, label='Intersection')], loc = 'lower center', bbox_to_anchor=(0.5, -0.125), frameon=False)
    return fig
model = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)

# 1. Create dataset #Note this is for testing
data_dir = Path('/work3/s211469/data')
patient_ids = np.loadtxt(data_dir / 'filenames_filtered.txt', dtype=str)
val_pct = 0.1
val_ids = np.random.choice(patient_ids, size=round(len(patient_ids)*val_pct), replace=False)
training_ids = [id for id in patient_ids if id not in val_ids]

# 1. Create dataset and validation set
train_set = BrainDataset(patient_ids=training_ids, data_dir=data_dir, binary='WT')
loader_args = dict(batch_size=1, num_workers=4)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
for batch in train_loader:
    images, true_masks, patient_ids = batch
    
    images = images.to(device=device, dtype=torch.float32)
    true_masks = true_masks.to(device=device, dtype=torch.long)

    mask_true_train = F.one_hot(true_masks[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
    mask_pred = model(images.to(device=device, dtype=torch.float32))
    mask_pred_train = np.argmax(mask_pred.detach().cpu().numpy(), axis=1)
    mask_pred_train = F.one_hot(torch.from_numpy(mask_pred_train[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
    img_train = images[0][0]
    patient_id = patient_ids[0]
    fig = predictions_plot(img_train, mask_true_train, mask_pred_train, patient_id=patient_id)
    break