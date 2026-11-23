import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import ratio


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def foreground_mask(logits):
    return torch.softmax(logits, dim=1).argmax(dim=1)


def focal_loss(logits, label, alpha=0.9, gamma=3.0):
    log_pt = F.log_softmax(logits, dim=1).gather(1, label.unsqueeze(1)).squeeze(1)
    weight = torch.where(label == 1, alpha, 1 - alpha)
    return (-weight * (1 - log_pt.exp()).pow(gamma) * log_pt).mean()


def create_lr_scheduler(optimizer, steps_per_epoch, epochs, warmup_epochs=10,
                        initial_lr=1e-7, final_lr=1e-6):
    if not 0 <= warmup_epochs < epochs or steps_per_epoch < 1:
        raise ValueError("Warm-up must be shorter than training, and the training loader must be non-empty.")
    if len(optimizer.param_groups) != 1:
        raise ValueError("The paper schedule expects one Adam parameter group.")
    peak_lr = optimizer.param_groups[0]["lr"]
    if not 0 < initial_lr <= peak_lr or not 0 < final_lr <= peak_lr:
        raise ValueError("Initial and final learning rates must be positive and at most the peak rate.")
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def multiplier(step):
        if step < warmup_steps:
            lr = initial_lr + (peak_lr - initial_lr) * step / warmup_steps
        else:
            fraction = (min(step, total_steps) - warmup_steps) / (total_steps - warmup_steps)
            lr = final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * fraction))
        return lr / peak_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def train_one_epoch(model, optimizer, scheduler, loader, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = focal_loss(model(images), labels) + 0.01 * model.orthogonality_loss()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach().item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def validation_f1(model, loader, device):
    model.eval()
    counts = torch.zeros(3, dtype=torch.int64, device=device)
    for images, labels in loader:
        labels = labels.to(device).bool()
        pred = foreground_mask(model(images.to(device))).bool()
        counts += torch.stack(((pred & labels).sum(), pred.sum(), labels.sum()))
    tp, positives_pred, positives_ref = counts.tolist()
    return ratio(2 * tp, positives_pred + positives_ref)
