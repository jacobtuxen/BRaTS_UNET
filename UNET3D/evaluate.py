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
    confusion = torch.zeros(net.n_classes, net.n_classes)

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