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
#from torchmetrics import JaccardIndex, ConfusionMatrix, Dice


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
    plt.box(False)
    ax.axis('off')
    return fig
#MAIN
# 1. Load data
data_dir = Path(str('/work3/s211469/data'))
val_ids = np.loadtxt(data_dir / 'val_ids.txt', dtype=str)

# 2. Create dataset for validation
val_set_WT = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='WT')
val_set_TC = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='TC')
val_set_MT = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='MT')

models_path = Path(str('/work3/s211469/models'))

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

#New plotting code
for val_idx in range(len(val_ids)):
    #Make directory for plots
    patient_dir = data_dir / 'Plots' / val_ids[val_idx]
    patient_dir.mkdir(parents=True, exist_ok=True)
    for plot_idx, model, val_set in zip(range(len(models)), models, val_sets):
        model.eval()

        image_val, true_masks_val, patient_ids_val = val_set[val_idx]   
        image_val = image_val.unsqueeze(0)
        true_masks_val = true_masks_val.unsqueeze(0)    
        
        mask_true_val = F.one_hot(true_masks_val[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
        mask_pred_val = model(image_val.to(device=device, dtype=torch.float32))
        mask_pred_val = np.argmax(mask_pred_val.detach().cpu().numpy(), axis=1)

        mask_pred_val = F.one_hot(torch.from_numpy(mask_pred_val[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
        img_val = image_val[0][0]
        patient_id_val = val_ids[val_idx]
        fig_val = predictions_plot(img_val, mask_true_val, mask_pred_val, patient_id=patient_id_val)
        fig_val.savefig(patient_dir / f'prediction_{models_names[plot_idx]}.png')










#predictions = [np.sum([model(val_patient).argmax(dim=1).cpu().numpy() for model in models],axis=0) for val_patient in val_ids]

# num_classes = 4
# jaccard = JaccardIndex(task="multiclass" ,num_classes=num_classes)
# dice = Dice(num_classes=num_classes)
# confusionmat = ConfusionMatrix(task = 'multiclass', num_classes=num_classes)

# jaccard_scores = [jaccard(torch.from_numpy(prediction), torch.from_numpy(true_mask)) for prediction, true_mask in zip(predictions, true_masks)]
# dice_scores = [dice(torch.from_numpy(prediction), torch.from_numpy(true_mask)) for prediction, true_mask in zip(predictions, true_masks)]
# confusion = [confusionmat(torch.from_numpy(prediction), torch.from_numpy(true_mask)) for prediction, true_mask in zip(predictions, true_masks)]
# #For testing
# # Make 3 predictions of flots size [batch,2,128,128]
# batch_size = 4
# output1 = torch.rand(batch_size,2,128,128,128)
# output2 = torch.rand(batch_size,2,128,128,128)
# output3 = torch.rand(batch_size,2,128,128,128)

# pred1 = output1.argmax(dim=1).cpu().numpy()
# pred2 = output2.argmax(dim=1).cpu().numpy()
# pred3 = output3.argmax(dim=1).cpu().numpy()

# print(f'pred1 shape: {pred1.shape}, {np.unique(pred1)}')
# print(f'pred2 shape: {pred2.shape}, {np.unique(pred2)}')
# print(f'pred3 shape: {pred3.shape}, {np.unique(pred3)}')

# total_predictions = np.sum([pred1, pred2, pred3], axis=0)
# print(f'total_predictions shape: {total_predictions.shape}, {np.unique(total_predictions)}')
