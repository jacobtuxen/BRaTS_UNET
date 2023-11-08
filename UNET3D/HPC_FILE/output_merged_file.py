# Content from: UNET3D/unet_model/unet_parts.py
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
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
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

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

# Content from: UNET3D/data_loader.py
import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from pathlib import Path

class BrainDataset(Dataset):
    def __init__(self, patient_ids: list, data_dir: Path):
        self.patient_ids = patient_ids
        self.data_dir = data_dir
        self.data = []

        self.init_data()

    def init_data(self):
        ids = ['flair.nii.gz','t1.nii.gz', 't1ce.nii.gz', 't2.nii.gz','seg.nii.gz']
        for patient_id in self.patient_ids:
            self.data.append([nib.load(self.data_dir / patient_id / f'{patient_id}_{id}').get_fdata() for id in ids])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        flair, t1, t1c ,t2, target = self.data[idx]
        t1 = torch.from_numpy(t1).float16()
        t1c = torch.from_numpy(t1c).float16()
        t2 = torch.from_numpy(t2).float16()
        flair = torch.from_numpy(flair).float16()
        target = torch.from_numpy(target).float16()

        #Normalize
        t1 = (t1 - t1.mean()) / t1.std()+1e-8
        t1c = (t1c - t1c.mean()) / t1c.std()+1e-8
        t2 = (t2 - t2.mean()) / t2.std()+1e-8
        flair = (flair - flair.mean()) / flair.std()+1e-8

        t1 = t1.unsqueeze(0)
        t1c = t1c.unsqueeze(0)
        t2 = t2.unsqueeze(0)
        flair = flair.unsqueeze(0)
        target = target.unsqueeze(0)

        #Input data size [B,C,W,D,H] 
        #Concatenate t1,t1c,t2,flair
        x = torch.concatenate((t1,t1c,t2,flair), dim=0)
        return x, target


# Content from: UNET3D/evaluate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm



@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            mask_true = torch.argmax(mask_true, dim=1)


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

# Content from: UNET3D/predictions.py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def plot_predictions(model, device, input, target):
    #input is (b,c,w,h,d)
    with torch.no_grad:
        pred = model(input)
    pred = pred.numpy()
    target = target.numpy()
    input = input.numpy()

    fig = plt.figure()
    plot_titles = ['t1', 't1c', 't2', 'flair', 'gt', 'pred']
    n_slices = 5
    slices = np.linspace(20,140, n_slices)
    for idx_i, slice in enumerate(slices):
        for idx_j, title in enumerate(plot_titles):
            plt.subplot(n_slices*len(plot_titles), idx_j % len(plot_titles + idx_i * len(plot_titles + 1)))
            if idx_j <= 3:
                plt.imshow(input[0,idx_j,:,:,slice])
            if idx_j == 4:
                plt.imshow(target[0,idx_j,:,:,slice])
            else:
                plt.imshow(pred[0,idx_j,:,:,slice])
            if idx_i == 0:
                plt.title(title)
    return fig



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
import numpy as np
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import ConcatDataset
from datetime import datetime
import time
import wandb

WANDB_API_KEY=""
USE_WANDB = False

def train_model(
        model,
        device,
        epochs: int = 5,
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
    if wandb_active:
        wandb.init(
            project=f"UNE3D_{timestamp}",
            config={
                "epochs": {epochs},
                "batch_size": {batch_size},
                "learning_rate": {learning_rate},
                "optimizer": {optimizer},
                "amp": {amp},
                "weight_decay": {weight_decay},
                "momentum": {momentum},
                "gradient_clipping": {gradient_clipping}
            }
        )
    

    # 1. Create dataset #Note this is for testing
    patient_ids = ['BraTS2021_00380']
    data_dir = Path.home() /'02456_Deep_Learning' / 'data'
    dataset = BrainDataset(patient_ids=patient_ids, data_dir=data_dir)
    train_set = dataset
    dataset_val = BrainDataset(patient_ids=patient_ids, data_dir=data_dir)
    val_set = dataset_val
    
    # 3. Create data loaders set numworkers=os.cpu_count() for faster data loading
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
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
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    all_epoch_losses = []
    all_accuracy = []
    total_training_time = 0
    for epoch in range(1, epochs + 1):
        print('epoch started')
        model.train()
        epoch_loss = 0
        for batch in train_loader:
              start_time = time.time()  # start timing
              images, true_masks = batch[0], batch[1]
              

              assert images.shape[1] == model.n_channels, \
                  f'Network has been defined with {model.n_channels} input channels, ' \
                  f'but loaded images have {images.shape[1]} channels. Please check that ' \
                  'the images are loaded correctly.'

              images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
              true_masks = true_masks.to(device=device, dtype=torch.long)

              with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                  masks_pred = model(images)
                  if model.n_classes == 1:
                      loss = criterion(masks_pred.squeeze(1), true_masks.float())
                      loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                  else:
                      true_masks = torch.argmax(true_masks, dim=1)
                    #   print(f"masks_pred shape: {masks_pred.shape}, dtype: {masks_pred.dtype}")
                    #   print(f"true_masks shape: {true_masks.shape}, dtype: {true_masks.dtype}")
                    #   print(f"Unique values in true_masks: {torch.unique(true_masks)}")
                      loss = criterion(masks_pred, true_masks)
                      loss += dice_loss(
                          F.softmax(masks_pred, dim=1).float(),
                          F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                          multiclass=True
                      )

              optimizer.zero_grad(set_to_none=True)
              grad_scaler.scale(loss).backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
              grad_scaler.step(optimizer)
              grad_scaler.update()

              global_step += 1
              epoch_loss += loss.item()
              all_epoch_losses.append(epoch_loss)

              #LOG WANDB
              if wandb_active:
                wandb.log({"train/train_loss": epoch_loss,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train/epoch_training_time": epoch_training_time,
                            "train/total_training_time": total_training_time,
                            "train/accuracy": val_score.item()
                            })

              epoch_training_time = time.time() - start_time  # end timing
              total_training_time += epoch_training_time

    val_score = evaluate(model, val_loader, device, amp)
    all_accuracy.append(val_score.item())
    if wandb_active:
        fig = plot_predictions(model, device, images, true_masks)
        wandb.log({"val/val_accuracy": val_score.item(),
        })
    scheduler.step(val_score)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
name_indentifier = f'UNet_BRaTS{timestamp}_'
dir_checkpoint = Path(f'./cp_{name_indentifier}/')
info_file = f'info_{name_indentifier}.txt'

#LOGIN
if USE_WANDB:
    wandb.login(key=WANDB_API_KEY)
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val/val_accuracy"},
        "parameters": {
            "batch_size": {"values": [16, 32, 64, 128]},
            "lr": {"max": 1e-3, "min": 1e-6},
            "epochs": {"values": [30,60,100]},
            "weight_decay": {"max": 1e-3, "min": 1e-6},
            "momentum": {"values": [0.9, 0.99]},
            "amp": {"values": [True, False]},
            "gradient_clipping": {"values": [0.1, 0.5, 1.0]},
            "optimzer": {"values": ["Adam", "RMSprop", "SGD"]},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UNET3D_SWEEP_{timestamp}")
model = UNet3D(n_channels=4, n_classes=3, bilinear=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
if USE_WANDB:
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
    wandb.agent(sweep_id, function=train_model, count=4)
else:
    train_model(model=model, device=device)
    print("Training done!")

