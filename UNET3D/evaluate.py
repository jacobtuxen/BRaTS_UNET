import torch
import torch.nn.functional as F
from monai.losses import *
from torchmetrics import JaccardIndex, Dice

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    jaccard_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in dataloader:
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long)
            jaccard = JaccardIndex(num_classes=net.n_classes)
            dice = Dice(num_classes=net.n_classes)


            # predict the mask
            mask_pred = net(image)
            mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 4, 1, 2, 3).float()
            mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 4, 1, 2, 3).float()
            # compute the Dice score, ignoring background
            jaccard_score += jaccard(mask_pred, mask_true)
            dice_score = dice(mask_true, mask_true)
            print(f"Jaccard Score: {jaccard_score}")
            print(f"Dice Score: {dice_score}")

    net.train()
    return dice_score / max(num_val_batches, 1)