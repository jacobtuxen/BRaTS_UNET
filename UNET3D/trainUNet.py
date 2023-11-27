
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
from evaluate import evaluate
from unet_model.unet_model import UNet3D
from monai.losses import *
import matplotlib.pyplot as plt
# from utils.dice_score import dice_loss
from UNET3D.plot import predictions_plot
from UNET3D.data_loader import BrainDataset
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
    data_dir = Path('/work3/s194572/data')
    patient_ids = np.loadtxt(data_dir / 'filenames.txt', dtype=str)
    val_pct = 0.1
    val_ids = np.random.choice(patient_ids, size=round(len(patient_ids)*val_pct), replace=False)
    training_ids = [id for id in patient_ids if id not in val_ids]

    # 1. Create dataset and validation set
    train_set = BrainDataset(patient_ids=training_ids, data_dir=data_dir, binary='WT')
    val_set = BrainDataset(patient_ids=val_ids, data_dir=data_dir, binary='WT')

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
                wandb.log({"train/train_loss": loss.item()})
        
        # 6. Evaluate the model on the validation set
        val_score_dice, val_score_jaccard, confusion = evaluate(model, val_loader, device, amp)
        scheduler.step(val_score_dice)
        if wandb_active:
            wandb.log({"val/val_accuracy_dice": val_score_dice})
            wandb.log({"val/val_accuarce_jaccard:": val_score_jaccard})
            wandb.log({"train/epoch_loss": epoch_loss/len(train_loader)})
            print(f"confusion: {confusion}, at epoch {epoch}")
            if epoch % 2 == 0:
                
                mask_true_train = F.one_hot(true_masks[0].unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
                mask_pred = model(images.to(device=device, dtype=torch.float32))
                mask_pred_train = np.argmax(mask_pred.detach().cpu().numpy(), axis=1)
                mask_pred_train = F.one_hot(torch.from_numpy(mask_pred_train[0]).unsqueeze(0), model.n_classes).permute(0, 4, 1, 2, 3).float()
                img_train = images[0][0]
                patient_id = patient_ids[0]
                fig = predictions_plot(img_train, mask_true_train, mask_pred_train, patient_id=patient_id)
                wandb.log({"train/plot": fig})
                fig.clf()
                plt.close(fig)
                
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
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=f"UNET3D_GDFL_{timestamp}")

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
                    wandb_active=True
                    )

    else:
        train_model(model=model, device=device)

if USE_WANDB:
    wandb.agent(sweep_id, function=run_model, count=10)
else:
    run_model()

print("Training done!")