# Content from: UNET3D/unet_model/unet_parts.py
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

# Content from: UNET3D/unet_model/unet_model.py
""" Full assembly of the parts to form the complete network """



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
            if self.device.type == 'cuda' and not NO_PRINT:
                print(f'Current memory allocated: {torch.cuda.memory_allocated(self.device)/1024**3:.2f} GB')
                print(f'Max memory allocated: {torch.cuda.max_memory_allocated(self.device)/1024**3:.2f} GB')
                print(f'Current memory cached: {torch.cuda.memory_reserved(self.device)/1024**3:.2f} GB')
                print(f'Max memory cached: {torch.cuda.max_memory_reserved(self.device)/1024**3:.2f} GB')

        print_memory_usage()
        x1 = self.inc(x)
        print("Memory usage after inc function:")
        print_memory_usage()

        x2 = self.down1(x1)
        print("Memory usage after down1 function:")
        print_memory_usage()

        x3 = self.down2(x2)
        print("Memory usage after down2 function:")
        print_memory_usage()

        x4 = self.down3(x3)
        print("Memory usage after down3 function:")
        print_memory_usage()

        x = self.up1(x4, x3)
        print("Memory usage after up1 function:")
        print_memory_usage()

        x = self.up2(x, x2)
        print("Memory usage after up2 function:")
        print_memory_usage()

        x = self.up3(x, x1)
        print("Memory usage after up3 function:")
        print_memory_usage()

        logits = self.outc(x)
        print("Memory usage after outc function:")
        print_memory_usage()

        return logits


# Content from: UNET3D/utils/dice_score.py
import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() >= 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

# Content from: UNET3D/utils/focal_loss.py
from typing import Optional, Sequence
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.

    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.

    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl

# Content from: UNET3D/plot.py
import nibabel as nib
from pathlib import Path
import torch
import numpy as np
from torch.nn.functional import one_hot
from skimage.util import montage
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET')
import torch.nn.functional as F
from matplotlib.lines import Line2D

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

# #Testing data,label,test_label is size 
# data = torch.load('data.pt')
# label = torch.load('label.pt')
# test_label = torch.load('test_label.pt')
# print(f'raw shapes: data: {data.shape}, label: {label.shape}, test_label: {test_label.shape}')

# temp_label = label[:,1:4,:,:,:]
# sum_label = torch.sum(temp_label, dim=1).unsqueeze(1)
# mask_true = torch.cat([label[:,0,:,:,:].unsqueeze(1), sum_label], dim=1)

# temp_label = test_label[:,1:4,:,:,:]
# sum_label = torch.sum(temp_label, dim=1).unsqueeze(1)
# mask_pred = torch.cat([test_label[:,0,:,:,:].unsqueeze(1), sum_label], dim=1)

# image = data.squeeze()[0]

# print(f'plotting shapes: {image.shape}, mask_true: {mask_true.shape}, mask_pred: {mask_pred.shape}')
# #Input shapes are image: [128, 128, 128]), mask_true:[1, 2, 128, 128, 128], mask_pred: [1, 2, 128, 128, 128]
# fig = predictions_plot(image, mask_true, mask_pred)

# plt.show()

# model = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device=device)
# print('loading model')
#Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path(str('/Users/christianvalentinkjaer/Documents/DTU/E23/02456_Deep_Learning/Brain_Project/BRaTS_UNET/data/archive'))
# dataset = BrainDataset(patient_ids, data_dir, binary='WT')
# data, target,_ = dataset[0]
# print(data.shape)
# print(target.shape)
# print(np.unique(target))
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
# print('loading data')
# for batch in train_loader:
#     images, true_masks, patient_ids = batch
#     # mask_true_train = F.one_hot(true_masks[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
#     mask_pred = model(images.to(device=device, dtype=torch.float32))
#     mask_pred = F.one_hot(mask_pred.argmax(dim=1), model.n_classes).permute(0, 4, 1, 2, 3).float()
#     print(f'mask_pred shape: {mask_pred.shape}')
#     print(f'mask_true shape: {true_masks.shape}')
#     # mask_pred_train = np.argmax(mask_pred.detach().cpu().numpy(), axis=1)
#     # mask_pred_train = F.one_hot(torch.from_numpy(mask_pred_train[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
#     # img_train = images[0][0]
#     # patient_id = patient_ids[0]
#     # # fig = predictions_plot(img_train, mask_true_train, mask_pred_train, patient_id=patient_id)
#     # # plt.show()
#     break

# Content from: UNET3D/utils/generalized_dice.py
import torch
import torch.nn as nn

class _AbstractDiceLoss(nn.Module):
    """
    Base class for different implementations of Dice loss.
    """

    def __init__(self, weight=None, normalization='sigmoid'):
        super(_AbstractDiceLoss, self).__init__()
        self.register_buffer('weight', weight)
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify `normalization=Softmax`
        assert normalization in ['sigmoid', 'softmax', 'none']
        if normalization == 'sigmoid':
            self.normalization = nn.Sigmoid()
        elif normalization == 'softmax':
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, input, target, weight):
        # actual Dice score computation; to be implemented by the subclass
        raise NotImplementedError

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        # compute per channel Dice coefficient
        per_channel_dice = self.dice(input, target, weight=self.weight)

        # average Dice score across all channels/classes
        return 1. - torch.mean(per_channel_dice)

class GeneralizedDiceLoss(_AbstractDiceLoss):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf.
    """

    def __init__(self, normalization='sigmoid', epsilon=1e-6):
        super().__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, input, target, weight):
        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        input = torch.flatten(input)
        target = torch.flatten(target)
        target = target.float()

        if input.size(0) == 1:
            # for GDL to make sense we need at least 2 channels (see https://arxiv.org/pdf/1707.03237.pdf)
            # put foreground and background voxels in separate channels
            input = torch.cat((input, 1 - input), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL weighting: the contribution of each label is corrected by the inverse of its volume
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False

        intersect = (input * target).sum(-1)
        intersect = intersect * w_l

        denominator = (input + target).sum(-1)
        denominator = (denominator * w_l).clamp(min=self.epsilon)

        return 2 * (intersect.sum() / denominator.sum())

# Content from: UNET3D/data_loader.py
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

# #Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path.home() / 'Desktop' / 'Deep Learning' / 'BRaTS_UNET' / 'data' / 'archive'
# dataset = BrainDataset(patient_ids, data_dir, binary='TC')
# data, target,_ = dataset[0]
# print(data.shape)
# print(target.shape)
# #visualize target
# import matplotlib.pyplot as plt
# plt.imshow(target[:,:,70])
# plt.show()

# Content from: UNET3D/evaluate.py
import torch
import torch.nn.functional as F
from monai.losses import *
from torchmetrics import JaccardIndex, ConfusionMatrix
from torchmetrics import Dice as DiceMetric
import torch.nn.functional as F

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    jaccard = JaccardIndex(task="binary" ,num_classes=net.n_classes).to(device=device)
    dice = DiceMetric(num_classes=net.n_classes).to(device=device)
    confusionmat = ConfusionMatrix(task = 'binary', num_classes=net.n_classes).to(device=device)
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
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 4, 1, 2, 3).long()

            # predict the mask
            mask_pred = net(image)
            # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 4, 1, 2, 3).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 4, 1, 2, 3).long()
            # compute the Dice score, ignoring background
            mask_pred = mask_pred.to(device=device)
            print(f'Mask true on device: {mask_true.device}')
            print(f'Mask pred on device: {mask_pred.device}')


            jaccard_score += jaccard(mask_pred, mask_true)
            dice_score += dice(mask_pred, mask_true)
            confusion += confusionmat(mask_pred, mask_true)

    net.train()
    return dice_score / max(num_val_batches, 1), jaccard_score / max(num_val_batches, 1), confusion

# Content from: UNET3D/trainUNet.py

import logging
from pathlib import Path
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.ops as tvops
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data import ConcatDataset
from datetime import datetime
from monai.losses import *
import matplotlib.pyplot as plt
import wandb
import gc

WANDB_API_KEY="fa06c10dd6495a8b9afda9eb0e328ab57f243479"
USE_WANDB = True

def train_model(
        model,
        device,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        optimizer: str = "RMSprop",
        wandb_active = False,
        dataset_type = 'WT'
):
    # 0. Set up loggin for wandb
    #setup wandb    

    # 1. Create dataset #Note this is for testing
    data_dir = Path('/work3/s211469/data')
    patient_ids = np.loadtxt(data_dir / 'filenames_filtered.txt', dtype=str)
    val_pct = 0.1
    val_ids = np.random.choice(patient_ids, size=round(len(patient_ids)*val_pct), replace=False)
    training_ids = [id for id in patient_ids if id not in val_ids]

    # 1. Create dataset and validation set
    train_set = BrainDataset(patient_ids=training_ids, data_dir=data_dir, binary=dataset_type)
    val_set = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary=dataset_type)

    # 3. Create data loaders set numworkers=4 as requested on HPC for faster data loading
    loader_args = dict(batch_size=batch_size, num_workers=4)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = GeneralizedDiceFocalLoss(softmax=True, gamma=2.0)
    global_step = 0
 

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch} started')
        model.train()
        epoch_loss = 0
        for batch in train_loader:
              images, true_masks, patient_ids = batch

              assert images.shape[1] == model.n_channels, \
                  f'Network has been defined with {model.n_channels} input channels, ' \
                  f'but loaded images have {images.shape[1]} channels. Please check that ' \
                  'the images are loaded correctly.'

              images = images.to(device=device, dtype=torch.float32)
              true_masks = true_masks.to(device=device, dtype=torch.long)

              with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                masks_pred = model(images)
                # loss = criterion(masks_pred, true_masks)
                loss = criterion(masks_pred, F.one_hot(true_masks, model.n_classes).permute(0, 4, 1, 2, 3).float())
                if wandb_active:
                    wandb.log({"train/train_loss": loss.item()})

              optimizer.zero_grad(set_to_none=True)
              grad_scaler.scale(loss).backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
              grad_scaler.step(optimizer)
              grad_scaler.update()

              global_step += 1
              epoch_loss += loss.item()
              if device.type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()

              #LOG WANDB
              if wandb_active:
                wandb.log({"train/epoch_loss": epoch_loss/len(train_loader)})
        
        # 6. Evaluate the model on the validation set
        val_score_dice, val_score_jaccard, confusion = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score_dice)
        if wandb_active:
            wandb.log({"val/val_accuracy_dice": val_score_dice})
            wandb.log({"val/val_accuarce_jaccard:": val_score_jaccard})
            wandb.log({"train/epoch_loss": epoch_loss/len(train_loader)})
            print(f"confusion: {confusion}, at epoch {epoch}")
                            
            mask_true_train = F.one_hot(true_masks[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
            mask_pred_train = np.argmax(masks_pred.detach().cpu().numpy(), axis=1)
            mask_pred_train = F.one_hot(torch.from_numpy(mask_pred_train[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
            img_train = images[0][0]
            patient_id = patient_ids[0]
            fig = predictions_plot(img_train, mask_true_train, mask_pred_train, patient_id=patient_id)
            wandb.log({"train/plot": fig})
            fig.clf()
            plt.close(fig)

            if epoch % 5 == 0:                
                model.eval()
                image_val, true_masks_val, patient_ids_val = next(iter(val_loader))
                mask_true_val = F.one_hot(true_masks_val[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
                mask_pred_val = model(image_val.to(device=device, dtype=torch.float32))
                mask_pred_val = np.argmax(mask_pred_val.detach().cpu().numpy(), axis=1)
                mask_pred_val = F.one_hot(torch.from_numpy(mask_pred_val[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
                img_val = image_val[0][0]
                patient_id_val = patient_ids_val[0]
                fig_val = predictions_plot(img_val, mask_true_val, mask_pred_val, patient_id=patient_id_val)
                wandb.log({"val/plot": fig_val})
                fig_val.clf()
                plt.close(fig_val)
                model.train()
#LOGIN
if USE_WANDB:
    timestamp = datetime.now().strftime("%Y%d%m-%H%M%S")
    wandb.login(key=WANDB_API_KEY)
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val/val_accuracy"},
        "parameters": {
            "batch_size": {"values": [6,8]},
            "lr": {"max": 1e-3, "min": 1e-6},
            "epochs": {"values": [50]},
            "weight_decay": {"max": 1e-3, "min": 1e-6},
            "momentum": {"values": [0.9, 0.99]},
            "amp": {"values": [True]},
            "gradient_clipping": {"values": [1.0]},
            "optimizer": {"values": ["Adam"]},
        }
    }
    #set dataloader
    dataset_type = ['WT', 'TC', 'MT']
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UNET3D_GDFL_{dataset_type[0]}_")

def run_model():
    model = UNet3D(n_channels=3, n_classes=2, trilinear=False, scale_channels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)
    if USE_WANDB:
        wandb.init(config=sweep_configuration)
        train_model(model=model, 
                    device=device, 
                    epochs=wandb.config.epochs, 
                    batch_size=wandb.config.batch_size, 
                    learning_rate=wandb.config.lr,
                    amp=wandb.config.amp,
                    weight_decay=wandb.config.weight_decay, 
                    momentum=wandb.config.momentum, 
                    gradient_clipping=wandb.config.gradient_clipping,
                    optimizer=wandb.config.optimizer,
                    wandb_active=True,
                    dataset_type=dataset_type[0]
                    )

    else:
        train_model(model=model, device=device)

if USE_WANDB:
    wandb.agent(sweep_id, function=run_model, count=10)
else:
    run_model()

print("Training done!")

