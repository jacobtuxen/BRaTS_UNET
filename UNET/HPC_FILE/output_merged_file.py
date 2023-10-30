# Content from: UNET/unet_model/unet_parts.py
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
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# Content from: UNET/unet_model/unet_model.py
""" Full assembly of the parts to form the complete network """



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
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

# Content from: UNET/utils/dice_score.py


# Content from: UNET/data_loader.py


# Content from: UNET/evaluate.py
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
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)

# Content from: UNET/trainUNet.py
import logging
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

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    train_set = dataset
    val_set = dataset_val
    
    # 3. Create data loaders set numworkers=os.cpu_count() for faster data loading
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    all_epoch_losses = []
    all_accuracy = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
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

    val_score = evaluate(model, val_loader, device, amp)
    all_accuracy.append(val_score.item())
    scheduler.step(val_score)
    if epoch%10 == 0:
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                torch.save(state_dict, str(dir_checkpoint / f'{name_indentifier}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')
    
    np.save(f'epoch_losses_{name_indentifier}.npy', all_epoch_losses)
    np.save(f'validation_losses_{name_indentifier}.npy', all_accuracy)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
name_indentifier = f'UNet_BRaTS{timestamp}_'
dir_checkpoint = Path(f'./cp_{name_indentifier}/')

model = UNet(n_channels=1, n_classes=3, bilinear=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
train_model(model, device=device, epochs=100, batch_size=32, learning_rate=1e-5, save_checkpoint=True, amp=False, weight_decay=1e-8, momentum=0.9, gradient_clipping=1.0)

