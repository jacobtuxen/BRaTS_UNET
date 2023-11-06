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
from evaluate import evaluate
from unet_model.unet_model import UNet3D
from utils.dice_score import dice_loss
import time
import wandb
WANDB_API_KEY=""

wandb.login(key=WANDB_API_KEY)

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
    # 0. Set up loggin for wandb
    #setup wandb
    wandb.init(
        project=f"UNET3D_{timestamp}",
        config={
            "epochs": {epochs},
            "batch_size": {batch_size},
            "learning_rate": {learning_rate},
        }
    )
    

    # 1. Create dataset
    train_set = dataset = 0
    val_set = dataset_val = 0
    
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
    total_training_time = 0
    wandb.watch(model, log="all")
    for epoch in range(1, epochs + 1):
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
              wandb.log({"train/train_loss": epoch_loss,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/epoch_training_time": epoch_training_time,
                        "train/total_training_time": total_training_time,
                        "train/batch_size": batch_size,
                        "train/weight_decay": weight_decay,
                        })

              epoch_training_time = time.time() - start_time  # end timing
              total_training_time += epoch_training_time

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
    training_time_str = f'Total training time: {total_training_time:.4f} seconds\n'
    hyperparameters_str = f'Hyperparameters:\nEpochs: {epochs}\nBatch size: {batch_size}\nLearning rate: {learning_rate}\nWeight decay: {weight_decay}\nMomentum: {momentum}\nGradient clipping: {gradient_clipping}\n'
    with open(info_file, 'w') as f:
        f.write(training_time_str)
        f.write(hyperparameters_str)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
name_indentifier = f'UNet_BRaTS{timestamp}_'
dir_checkpoint = Path(f'./cp_{name_indentifier}/')
info_file = f'info_{name_indentifier}.txt'

model = UNet3D(n_channels=1, n_classes=3, bilinear=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device=device)
train_model(model, device=device, epochs=100, batch_size=32, learning_rate=1e-5, save_checkpoint=True, amp=False, weight_decay=1e-8, momentum=0.9, gradient_clipping=1.0)

