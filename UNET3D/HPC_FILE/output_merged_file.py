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

# Content from: UNET3D/visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def visualize_model_output(epoch, input, model, patient_id, device):
  slices = np.linspace(10,100, 5)
  plot_titles = ['flair', 't1ce', 't2','seg', 'output']

  model.eval()
  with torch.no_grad():
    input = input.unsqueeze(0).to(device)
    output = model(input).cpu().numpy()
    output = np.argmax(output, axis=1) #<- #0 air
    input_images = input.cpu().numpy()

  input_images = np.squeeze(input_images, axis=0)
  fig, ax = plt.subplots(len(slices), 5, figsize=(15, 5))
  seg = [nib.load(f'/work3/s194572/data/{patient_id}/{patient_id}_{titles}').get_fdata() for titles in ['seg.nii']]

  for i, slice in enumerate(slices):
    for j in range(len(plot_titles)):
      if j == 0:
        ax[i, j].imshow(input_images[0, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 1:
        ax[i, j].imshow(input_images[1, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 2:
        ax[i, j].imshow(input_images[2, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 3:
        ax[i, j].imshow(seg[0][:, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      elif j == 4:
        ax[i, j].imshow(output[0, :, :, int(slice)], cmap='gray')
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
      if i==0:
        ax[i, j].set_title(f'{plot_titles[j]}')
      
  model.train()
  return fig

# Content from: UNET3D/class_distributions.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

data_dir = Path('/work3/s211469/data')
patient_ids = np.loadtxt(data_dir / 'filenames.txt', dtype=str)
extension = 'seg.nii'
distributions = []

for patient_id in patient_ids:
    data_path = data_dir / patient_id / f'{patient_id}_{extension}'
    target = nib.load(data_path).get_fdata()
    
    start_idx = 56
    end_idx = 184
    start_idx_height = 13
    end_idx_height = 141
    target = target[start_idx:end_idx,start_idx:end_idx,start_idx_height:end_idx_height]

    target = np.where(target==4, 3, target)

    distribution = np.bincount(target.flatten().astype(int))
    distributions.append(distribution)
    print(f'Patient {patient_id} done')
#Save distributions
distributions = np.asarray(distributions)
np.save('class_distributions.npy', distributions)

# Content from: UNET3D/data_loader.py
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path
import torch.nn.functional as F

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
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
        #Cat
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
# dataset = BrainDataset(patient_ids, data_dir)
# data, target,_ = dataset[0]
# print(data.shape)
# print(target.shape)

# Content from: UNET3D/evaluate.py
import torch
import torch.nn.functional as F



@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)


            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 4, 1, 2, 3).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 4, 1, 2, 3).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

# Content from: UNET3D/trainUNet.py

import logging
from pathlib import Path
import sys
git_dir = Path.home() / 'Documents' / 'DTU' / 'E23' / '02456_Deep_Learning' / 'Brain_Project' / 'BRaTS_UNET'
sys.path.append(str(git_dir))
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
import matplotlib.pyplot as plt
from monai.losses import GeneralizedDiceLoss, GeneralizedDiceFocalLoss
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
):
    # 0. Set up loggin for wandb
    #setup wandb    

    # 1. Create dataset #Note this is for testing
    data_dir = Path('/work3/s211469/data')
    patient_ids = np.loadtxt(data_dir / 'filenames.txt', dtype=str)
    val_pct = 0.1
    val_ids = np.random.choice(patient_ids, size=round(len(patient_ids)*val_pct), replace=False)
    training_ids = [id for id in patient_ids if id not in val_ids]

    # 1. Create dataset and validation set
    train_set = BrainDataset(patient_ids=training_ids, data_dir=data_dir)
    val_set = BrainDataset(patient_ids=val_ids, data_dir=data_dir)

    # 3. Create data loaders set numworkers=4 as requested on HPC for faster data loading
    loader_args = dict(batch_size=batch_size, num_workers=4)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    match optimizer:
        case "Adam":
            optimizer = optim.Adam(model.parameters(),
                                    lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
        case "RMSprop":
            optimizer = optim.RMSprop(model.parameters(),lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        case "SGD":
            optimizer = optim.SGD(model.parameters(),lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # weights = torch.tensor([1.01553413, 517.7032716, 98.72168775, 309.07898017]).to(device)
    #criterion = nn.CrossEntropyLoss(weight=weights) <- legacy code maybe?
    criterion = GeneralizedDiceFocalLoss(to_onehot_y = True, softmax = True)
    val_criterion = GeneralizedDiceLoss(to_onehot_y = True, softmax = True)
    
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
                loss = criterion(masks_pred, true_masks.unsqueeze(1)) #Input must be (N, C, H, W, D)


                # loss = tvops.focal_loss.sigmoid_focal_loss(inputs=masks_pred, targets=F.one_hot(true_masks, model.n_classes).permute(0, 4, 1, 2, 3).float(), gamma=2.0, alpha=0.25, reduction='mean')
                # if wandb_active:
                #     wandb.log({"train/focal_loss": loss.item()})
                # loss += dice_loss(
                #     F.softmax(masks_pred, dim=1).float(),
                #     F.one_hot(true_masks, model.n_classes).permute(0, 4, 1, 2, 3).float(),
                #     multiclass=True
                # )

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
                wandb.log({"train/train_loss": loss.item()})
        
        # 6. Evaluate the model on the validation set
        model.eval()
        val_score = 0
        for val_batch in val_loader:
            images, true_masks, patient_ids = val_batch
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            masks_pred = model(images)
            loss = val_criterion(masks_pred, true_masks.unsqueeze(1))
            val_score += 1 - loss.item()
        val_score = val_score/len(val_loader)
        model.train()
        scheduler.step(val_score)

        # val_score = evaluate(model, val_loader, device, amp)
        # scheduler.step(val_score)
        
        if wandb_active:
            wandb.log({"val/val_accuracy": val_score})
            wandb.log({"train/epoch_loss": epoch_loss/len(train_loader)})
            if epoch % 5 == 0:
                fig = visualize_model_output(epoch, images[0], model, patient_ids[0], device)
                wandb.log({"train/plot": fig})
                fig.clf()
                plt.close(fig)
                image_val, _, patient_ids_val = next(iter(val_loader))
                image_val = image_val.to(device=device, dtype=torch.float32)
                fig_val = visualize_model_output(epoch, image_val[0], model, patient_ids_val[0], device)
                wandb.log({"val/plot": fig_val})
                fig_val.clf()
                plt.close(fig_val)
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
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UNET3D_focal_GDL_{timestamp}")

def run_model():
    model = UNet3D(n_channels=3, n_classes=4, trilinear=False, scale_channels=1)
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
                    wandb_active=True
                    )

    else:
        train_model(model=model, device=device)

if USE_WANDB:
    wandb.agent(sweep_id, function=run_model, count=10)
else:
    run_model()

print("Training done!")

