
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
from evaluate import evaluate
from unet_model.unet_model import UNet3D
from utils.dice_score import dice_loss
from UNET3D.visualize import visualize_model_output
from UNET3D.data_loader import BrainDataset
import time
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
    model = UNet3D(n_channels=4, n_classes=4, trilinear=True)
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