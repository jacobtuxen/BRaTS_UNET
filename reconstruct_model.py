import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import pywt
from skimage.util import montage
import scipy.stats as st


import torch
import torch.nn.functional as F
from monai.losses import *
from torchmetrics import JaccardIndex, ConfusionMatrix
from torchmetrics import Dice as DiceMetric
import torch.nn.functional as F

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    jaccard = JaccardIndex(task="multiclass" ,num_classes=net.n_classes).to(device=device)
    dice = DiceMetric(num_classes=net.n_classes).to(device=device)
    # confusion = ConfusionMatrix(num_classes=net.n_classes).to(device=device)
    num_val_batches = len(dataloader)
    dice_score = 0
    jaccard_score = 0
    confusion = torch.zeros(net.n_classes, net.n_classes).to(device=device)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            #mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 4, 1, 2, 3).long()

            # predict the mask
            mask_pred = net(image)
            # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 4, 1, 2, 3).float()
            #mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 4, 1, 2, 3).long()
            # compute the Dice score, ignoring background
            mask_pred = mask_pred.to(device=device)

            jaccard_score += jaccard(mask_pred, mask_true)
            dice_score += dice(mask_pred, mask_true)
            #confusion += confusionmat(mask_pred, mask_true)

    net.train()
    return dice_score / max(num_val_batches, 1), jaccard_score / max(num_val_batches, 1)

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path, binary='WT', reconstruction_wavelet = 'haar', threshold = 0): #(WT, TC, MT)
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.binary = binary
        self.extensions = ['flair.nii', 't1ce.nii', 't2.nii','seg.nii']
        self.wavelet = reconstruction_wavelet
        self.keys = ['aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']
        self.threshold_percentile = threshold

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.patient_ids)
    
    def wavelet_reconstruction(self, data):
        #make wavelet transform, then threshold, then inverse wavelet transform
        #data is a 4D tensor
        data = data.numpy()
        reconstructed_data = []
        for i in range(data.shape[0]):
            coeffs_of_levels = pywt.wavedecn(data[i], self.wavelet, level=4)
            temp = []
            for idx, coeffs in enumerate(coeffs_of_levels):
                for idxj, key in enumerate(self.keys):
                    if idx != 0:
                        temp.append(coeffs[key].reshape((1,-1)))
                    else:
                        temp.append(coeffs[0].reshape((1,-1)))
            temp = np.concatenate(temp, axis=1)
            threshold = np.percentile(np.abs(temp), self.threshold_percentile)
            count_zeroes = 0
            for idx, coeffs in enumerate(coeffs_of_levels):
                for idxj, key in enumerate(self.keys):
                    if idx != 0:
                        coeffs[key] = np.where(np.abs(coeffs[key]) < threshold, 0, coeffs[key])
                        count_zeroes += np.where(coeffs[key]==0, 1, 0).sum()
                    else:
                        coeffs[0] = np.where(np.abs(coeffs[0]) < threshold, 0, coeffs[0])
                        count_zeroes += np.where(coeffs[0]==0, 1, 0).sum()
            reconstructed_data.append(pywt.waverecn(coeffs_of_levels, self.wavelet))
            print(f'Percentage of zeroes: {count_zeroes/temp.shape[1]}')
        reconstructed_data = np.stack(reconstructed_data, axis=0)

        return torch.from_numpy(reconstructed_data)


    
    
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
        data = self.wavelet_reconstruction(data)
        target = target[start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]

        #normalize data in each channel min max normalization
        for i in range(data.shape[0]):
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())

        return data, target, patient_id
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        def print_memory_usage():
            NO_PRINT = False
            if self.device.type == 'cuda' and NO_PRINT:
                print(f'Current memory allocated: {torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB')
                print(f'Max memory allocated: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f} GB')
                print(f'Current memory cached: {torch.cuda.memory_reserved(self.device)/1024**3:.2f} GB')
                print(f'Max memory cached: {torch.cuda.max_memory_reserved(self.device)/1024**3:.2f} GB')
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

timestamps = ['20231112-204742',
              '20231112-232207',
              '20231212-101511',
              '20231112-203050',
              '20231112-223157',
              '20231112-232214',
              '20231112-203103',
              '20231112-224433',
              '20231212-101509']

thresholds = [0, 25, 50, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
wavelet = 'db3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for timestamp in timestamps:
    model_dir = '/work3/s194572/models/baseline'
    model = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
    model.load_state_dict(torch.load(f"{model_dir}/model_baseline_{timestamp}.pt"))
    model = model.to(device=device)
    study_no = 's194572'
    data_dir = Path(f'/work3/{study_no}/data')
    thres_dir = Path(f'/work3/{study_no}/models/baseline/npy')
    val_ids = np.loadtxt(f'/work3/s194572/models/baseline/filenames/{timestamp}val_ids.txt', dtype=str)

    
    for threshold in thresholds:
      val_set = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='WT', reconstruction_wavelet=wavelet, threshold=threshold)
      val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
      all_jaccard = []
      all_dice = []

      for val_idx, batch in enumerate(val_loader):
          images, true_masks, patient_ids_val = batch
          images = images.to(device=device, dtype=torch.float32)
          true_masks = true_masks.to(device=device, dtype=torch.long)
          masks_pred = model(images)
          val_score_dice, val_score_jaccard = evaluate(model, val_loader, device, False)


      all_jaccard = np.array(all_jaccard)
      all_dice = np.array(all_dice)
      print(all_jaccard, threshold)
      np.savetxt(thres_dir / f'all_jaccard_{wavelet}_thres_{threshold}.txt', all_jaccard)
      np.savetxt(thres_dir / f'all_dice_{wavelet}_thres_{threshold}.txt', all_dice)
      #Calculate CI intervals
      conf_ints_jac_all = st.t.interval(0.95, len(all_jaccard.flatten())-1, loc=np.mean(all_jaccard.flatten()), scale=st.sem(all_jaccard.flatten()))
      conf_ints_dice_all = st.t.interval(0.95, len(all_jaccard.flatten())-1, loc=np.mean(all_jaccard.flatten()), scale=st.sem(all_jaccard.flatten()))

      np.savetxt(thres_dir / f'conf_ints_jac_all_{wavelet}_thres_{threshold}.txt', conf_ints_jac_all)
      np.savetxt(thres_dir / f'conf_ints_dice_all_{wavelet}_thres_{threshold}.txt', conf_ints_dice_all)