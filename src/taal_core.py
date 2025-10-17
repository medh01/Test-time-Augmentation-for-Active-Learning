import torch
import torch.nn.functional as F
from typing import List

from augmentation_utils import augment_image, reverse_augmentations

#################### Jensen-Shannon Divergence (JSD) ####################
def entropy(probs: torch.Tensor, dim: int, eps: float=1e-8) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + eps), dim=dim)

def entropy_of_average_batch(
    prob_maps: torch.Tensor
) -> torch.Tensor:
    # prob_maps: [K, B, C, H, W]
    M = prob_maps.mean(dim=0)         # [B, C, H, W]
    return entropy(M, dim=1)          # [B, H, W]

def average_entropy_batch(
    prob_maps: torch.Tensor
) -> torch.Tensor:
    # prob_maps: [K, B, C, H, W]
    H_each = entropy(prob_maps, dim=2)  # sum over C → [K, B, H, W]
    return H_each.mean(dim=0)           # average over K → [B, H, W]

def JSD_batch(
    logits_list: List[torch.Tensor],
    alpha: float = 0.5
) -> torch.Tensor:
    """
    logits_list: list of K tensors, each [B, C, H, W]
    returns: [B] JSD scalar per image in the batch
    """
    # 1) Stack into [K, B, C, H, W]
    prob_maps = torch.stack([
        F.softmax(logits, dim=1)  # softmax over C
        for logits in logits_list
    ], dim=0)

    # 2) Compute entropies [B, H, W]
    H_M   = entropy_of_average_batch(prob_maps)
    H_avg = average_entropy_batch   (prob_maps)

    # 3) JSD map [B, H, W]
    jsd_map = alpha * H_M - (1.0 - alpha) * H_avg

    # 4) Reduce spatially to get one scalar per batch element
    return jsd_map.mean(dim=[1,2])   # → [B]



#################### Augment Predict Reverse Function ####################
def augment_predict_reverse(
    model: torch.nn.Module,
    images: torch.Tensor,        # [B, C_in, H, W] (can be on CPU or GPU)
    K: int = 3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    for_training: bool = True     # True → train consistency (grads ON), False → eval/AL scoring (grads OFF)
) -> torch.Tensor:
    B, C_in, H, W = images.shape
    model = model.to(device)

    # TAAL behavior:
    #  - training consistency: train mode, gradients enabled
    #  - AL/TTA scoring: eval mode, no gradients
    model.train() if for_training else model.eval()

    all_rev_logits = []
    ctx = torch.set_grad_enabled(for_training)
    with ctx:
        for _ in range(K):
            batch_aug, batch_params = [], []

            # Do augmentations on CPU (safer for ColorJitter/Blur), then move once to device
            for img in images:
                img_cpu = img.detach().to("cpu")                 # ensure CPU for torchvision augs
                img_aug, params = augment_image(img_cpu)         # your function
                batch_aug.append(img_aug)
                batch_params.append(params)

            batch_aug = torch.stack(batch_aug, dim=0).to(device, non_blocking=True)  # [B,C_in,H,W]
            logits = model(batch_aug)                                                # [B,C_out,H,W]

            # reverse per-sample geometry on logits
            rev_logits = []
            for b in range(B):
                rev = reverse_augmentations(
                    logits[b],
                    batch_params[b],
                    flip_axis=2,   # W in CHW
                    rot_axis0=1,   # H
                    rot_axis1=2    # W
                )  # [C_out,H,W]
                rev_logits.append(rev)
            rev_logits = torch.stack(rev_logits, dim=0)  # [B,C_out,H,W]
            all_rev_logits.append(rev_logits)

    # stack & permute → [B,K,C_out,H,W]
    all_rev_logits = torch.stack(all_rev_logits, dim=0)             # [K,B,C_out,H,W]
    all_rev_logits = all_rev_logits.permute(1,0,2,3,4).contiguous() # [B,K,C_out,H,W]
    return all_rev_logits

#################### JSD Consistency Loss ####################

def jsd_consistency_batch(
        model: torch.nn.Module,
        imgs_U: torch.Tensor,
        K: int = 3,
        alpha: float = 0.5,
        device: str = "cuda"
) -> torch.Tensor:  # scalar

    logits_U = augment_predict_reverse(model, imgs_U, K=K, device=device)  # [B,K,C,H,W]
    logits_list = [logits_U[:, k] for k in range(K)]                       # K × [B,C,H,W]
    loss_per_img = JSD_batch(logits_list, alpha=alpha)                     # [B]
    return loss_per_img.mean()                                            # scalar

#################### JSD Scoring Test Time Augmentation ####################

@torch.no_grad()
def jsd_score_tta(
    model: torch.nn.Module,
    imgs: torch.Tensor,          # [B,C,H,W]
    K: int = 3,
    alpha: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    TAAL acquisition scoring (eval mode, no grad).
    Returns per-image JSD_α scores: [B]
    """
    # keep imgs on CPU so your augment_image (ColorJitter/Blur) stays happy
    if imgs.is_cuda:
        imgs = imgs.detach().cpu()

    # eval + no-grad path inside (for_training=False)
    logits_tta = augment_predict_reverse(
        model, imgs, K=K, device=device, for_training=False
    )  # [B,K,C,H,W]

    logits_list = [logits_tta[:, k] for k in range(K)]  # K × [B,C,H,W]
    return JSD_batch(logits_list, alpha=alpha)          # [B]
