from itertools import cycle
from typing import Callable, Optional, Dict
from dataclasses import dataclass


import torch, torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from metrics import pixel_accuracy, macro_dice

@dataclass(frozen=True)
class SSLConfig:
    loss: Callable
    K: int = 3
    alpha: float = 0.5
    use: bool = True


def train_one_epoch(
    labeled_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_sup_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda",
    num_classes: int = 5,
    scaler: Optional[GradScaler] = None,
    scheduler=None,
    step_scheduler_on_batch: bool = False,
    # ---- SSL (optional) ------------------------------------
    unlabeled_loader=None,                  # pass only if using SSL
    ssl: Optional[SSLConfig] = None,        # config for the SSL loss
    ssl_weight: float = 0.0,                # λ (ramp per epoch outside)
) -> Dict[str, float]:
    """
    One training epoch that supports fully-supervised and semi-supervised training.

    Supervised: leave `ssl=None` or `ssl_weight=0` or `unlabeled_loader=None`.
    Semi-supervised: pass `unlabeled_loader`, `ssl=SSLConfig(...)`, and `ssl_weight>0`.
    """
    device = torch.device(device)
    scaler = scaler or GradScaler()
    model.train()

    use_ssl = (
        ssl is not None
        and ssl.use
        and ssl_weight > 0.0
        and (unlabeled_loader is not None)
        and callable(ssl.loss)
    )

    ul_iter = cycle(unlabeled_loader) if use_ssl else None

    tot_loss = tot_acc = tot_dice = 0.0
    n_batches = 0

    pbar = tqdm(labeled_loader, desc="Train", leave=False)
    for imgs_L, masks_L, *_ in pbar:
        imgs_L  = imgs_L.float().to(device, non_blocking=True)
        masks_L = masks_L.long().to(device,  non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast():
            # supervised branch
            logits_L = model(imgs_L)
            loss_sup = loss_sup_fn(logits_L, masks_L)

            # optional SSL branch
            loss_uns = torch.zeros((), device=device)
            if use_ssl:
                imgs_U, *_ = next(ul_iter)
                imgs_U = imgs_U.float().to(device, non_blocking=True)
                loss_uns = ssl.loss(model, imgs_U, K=ssl.K, alpha=ssl.alpha, device=device)

            loss = loss_sup + (ssl_weight * loss_uns)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler and step_scheduler_on_batch:
            scheduler.step()

        # metrics on the labeled stream
        with torch.no_grad():
            preds      = logits_L.argmax(1)
            batch_acc  = pixel_accuracy(preds, masks_L)
            batch_dice = macro_dice(preds, masks_L, num_classes)

        n_batches += 1
        tot_loss  += loss.item()
        tot_acc   += batch_acc.item()
        tot_dice  += batch_dice.item()

        pbar.set_postfix({
            "sup":  loss_sup.item(),
            "uns":  loss_uns.item() if use_ssl else 0.0,
            "λ":    ssl_weight if use_ssl else 0.0,
            "loss": tot_loss / n_batches,
            "dice": tot_dice / n_batches,
            "lr":   optimizer.param_groups[0]["lr"],
        })
    pbar.close()

    if n_batches == 0:
        raise ValueError("Labeled loader is empty – nothing to train on.")

    # Step epoch-based schedulers here (not ReduceLROnPlateau)
    if scheduler and not step_scheduler_on_batch and not isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
    ):
        scheduler.step()

    return {
        "loss": tot_loss / n_batches,
        "acc%": (tot_acc / n_batches) * 100.0,
        "dice": tot_dice / n_batches,
    }


@torch.no_grad()
def evaluate_loader(loader, model, device="cuda", num_classes=5):
    """Evaluates the model on a given data loader.

    This function iterates over the data, makes predictions, and calculates evaluation metrics.
    It operates in `torch.no_grad()` mode to save memory and computation.

    Args:
        loader (DataLoader): The data loader for the evaluation set.
        model (nn.Module): The model to be evaluated.
        device (str, optional): The device to evaluate on. Defaults to "cuda".
        num_classes (int, optional): The number of classes for metric calculation. Defaults to 4.

    Returns:
        tuple: A tuple containing the mean accuracy and mean Dice coefficient.
    """
    model.eval()  # inference mode
    device = torch.device(device)

    tot_acc, tot_dice, n_batches = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="Eval", leave=False)
    for imgs, masks, *_ in pbar:
        imgs = imgs.float().to(device, non_blocking=True)
        masks = masks.long().to(device, non_blocking=True)

        logits = model(imgs)  # (N, C, H, W)
        preds = logits.argmax(1)  # (N, H, W)

        batch_acc = pixel_accuracy(preds, masks)
        batch_dice = macro_dice(preds, masks, num_classes)

        # accumulate
        tot_acc += batch_acc.item()
        tot_dice += batch_dice.item()
        n_batches += 1

        pbar.set_postfix({"acc%": tot_acc / n_batches,
                          "dice": tot_dice / n_batches})
    pbar.close()

    if n_batches == 0:
        raise ValueError("Loader is empty – nothing to evaluate.")

    return tot_acc / n_batches, tot_dice / n_batches