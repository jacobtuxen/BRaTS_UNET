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

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
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
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if trilinear else 1
        self.up1 = (Up(256, 128 // factor, trilinear))
        self.up2 = (Up(128, 64 // factor, trilinear))
        self.up3 = (Up(64, 32, trilinear))
        self.outc = (OutConv(32, n_classes))

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

# Content from: UNET3D/visualize.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

def visualize_model_output(epoch, input, model, patient_id, device):
  slices = np.linspace(10,150, 5)
  plot_titles = ['flair','t1', 't1ce', 't2','seg']

  model.eval()
  with torch.no_grad():
    input = input.squeeze(0).to(device)
    output = model(input)
  
  fig, ax = plt.subplots(len(slices), 6, figsize=(15, 5))
  originals = [nib.load(f'/work3/s211469/data/{patient_id}/{patient_id}_{titles}').get_fdata() for titles in ['flair.nii','t1.nii', 't1ce.nii', 't2.nii','seg.nii']]
  for idj, slice_ in enumerate(slices):
    for idx, original in enumerate(originals):
      ax[idj, idx].imshow(original[:,:,slice_], cmap='gray')
      ax[idj, idx].axis('off')
      if idj == 0 and idx == 0:
        ax[0, idx].set_title(f'E: {epoch}, {plot_titles[idx]}')
      elif idj == 0:
        ax[0, idx].set_title(f'{plot_titles[idx]}')
    ax[idj, 5].imshow(output[:,:,slice_], cmap='gray')
    ax[idj, 5].axis('off')
    if idj == 0:
      ax[0, 5].set_title(f'Prediction')
  model.train()
  return fig

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

    def load_nifti_file(self, file_path):
        return nib.load(file_path).get_fdata()

    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data_paths = [self.data_dir / patient_id / f'{patient_id}_{data_id}' for data_id in ['flair.nii','t1.nii', 't1ce.nii', 't2.nii','seg.nii']]
        data = [self.load_nifti_file(path) for path in data_paths]
        target = torch.from_numpy(np.where(data[4]==4, 3, data[4])).long()
        #Cat
        data = torch.cat([torch.from_numpy(data[i]).unsqueeze(0) for i in range(4)], dim=0)

        start_idx = (data.shape[1]-160)//2
        end_idx = (data.shape[1]+160)//2

        data = F.pad(data, (0, 160 - data.shape[3], 0, 0)) 
        target = F.pad(target, (0, 160 - target.shape[2], 0, 0, 0, 0))


        
        data = data[:,start_idx:end_idx,start_idx:end_idx,:]
        target = target[start_idx:end_idx,start_idx:end_idx,:]
        
        #Normalize
        data = F.normalize(data, p=2, dim=0)
        

        return data, target, patient_id

#Test loader    
# patient_ids = ['BraTS2021_00495']
# data_dir = Path.home() / 'Documents' / 'DTU' / 'E23' / '02456_Deep_Learning' / 'Brain_Project' / 'BRaTS_UNET' / 'data' / 'archive'
# dataset = BrainDataset(patient_ids, data_dir)
# data, target,_ = dataset[0]
# print(data.shape)
# print(target.shape)

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
import gc

WANDB_API_KEY="fa06c10dd6495a8b9afda9eb0e328ab57f243479"
USE_WANDB = False

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
    weights = torch.tensor([1.0, 18134.2673, 122.652088, 575.141447]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        gc.collect()

    # 5. Begin training
    all_epoch_losses = []
    all_accuracy = []
    total_training_time = 0
    for epoch in range(1, epochs + 1):
        print('epoch started')
        if device.type == 'cuda':
            print(f'Current memory allocated: {torch.cuda.memory_allocated(device)/1024**3:.2f} GB')
            print(f'Max memory allocated: {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB')
            print(f'Current memory cached: {torch.cuda.memory_reserved(device)/1024**3:.2f} GB')
            print(f'Max memory cached: {torch.cuda.max_memory_reserved(device)/1024**3:.2f} GB')
        model.train()
        epoch_loss = 0
        for batch in train_loader:
              start_time = time.time()  # start timing
              images, true_masks, patient_ids = batch

              assert images.shape[1] == model.n_channels, \
                  f'Network has been defined with {model.n_channels} input channels, ' \
                  f'but loaded images have {images.shape[1]} channels. Please check that ' \
                  'the images are loaded correctly.'

              images = images.to(device=device, dtype=torch.float32)
              true_masks = true_masks.to(device=device, dtype=torch.long).unsqueeze(1)

              with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                  masks_pred = model(images)
                  if model.n_classes == 1:
                      loss = criterion(masks_pred.squeeze(1), true_masks.float())
                      loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                  else:
                      true_masks = torch.argmax(true_masks, dim=1)
                      print(f"masks_pred shape: {masks_pred.shape}, dtype: {masks_pred.dtype}")
                      print(f"true_masks shape: {true_masks.shape}, dtype: {true_masks.dtype}")
                      print(f"Unique values in true_masks: {torch.unique(true_masks)}")
                      loss = criterion(masks_pred, true_masks)
                      loss += dice_loss(
                          F.softmax(masks_pred, dim=1).float(),
                          F.one_hot(true_masks, model.n_classes).permute(0, 4, 1, 2, 3).float(),
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
    
    if epoch % 1 == 0:
            if USE_WANDB:
                fig = visualize_model_output(epoch, images[0], model, patient_ids[0], device)
                wandb.log({
                    "train/plot": fig,
            })
    val_score = evaluate(model, val_loader, device, amp)
    all_accuracy.append(val_score.item())
    scheduler.step(val_score)

#LOGIN
if USE_WANDB:
    timestamp = datetime.now().strftime("%Y%d%m-%H%M%S")
    wandb.login(key=WANDB_API_KEY)
    sweep_configuration = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "val/val_accuracy"},
        "parameters": {
            "batch_size": {"values": [1,2,4]},
            "lr": {"max": 1e-3, "min": 1e-6},
            "epochs": {"values": [30,60,100]},
            "weight_decay": {"max": 1e-3, "min": 1e-6},
            "momentum": {"values": [0.9, 0.99]},
            "amp": {"values": [True, False]},
            "gradient_clipping": {"values": [0.1, 0.5, 1.0]},
            "optimzer": {"values": ["RMSprop"]},
        }
    }
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UNET3D_SWEEP_{timestamp}")

def run_model():
    model = UNet3D(n_channels=4, n_classes=4, trilinear=False)
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
    wandb.agent(sweep_id, function=run_model, count=20)
else:
    run_model()

print("Training done!")

