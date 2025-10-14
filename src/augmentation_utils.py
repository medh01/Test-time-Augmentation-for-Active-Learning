import random
import torch
import torchvision.transforms as T

def add_gaussian_noise(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0
) -> torch.Tensor:
    noise = torch.randn_like(tensor) * std + mean
    return (tensor + noise).clamp(0.0, 1.0)


def apply_geometric(x, aug_dict, flip_axis, rot_axis0, rot_axis1):
    if aug_dict["flip"]:
        x = torch.flip(x, [flip_axis])
    if aug_dict["rot"]:
        x = torch.rot90(x, aug_dict["rot"], dims=(rot_axis0,rot_axis1))
    return x

def apply_photometric(x, aug_dict):
    if aug_dict["jitter"] > 0:
        jitter_tf = T.ColorJitter(
            brightness=aug_dict["jitter"],
            contrast=aug_dict["jitter"],
            saturation=aug_dict["jitter"],
            hue=aug_dict["jitter"]
        )
        x = jitter_tf(x)
    if aug_dict["blur"]   > 1:
        blur_tf = T.GaussianBlur(
            kernel_size=(aug_dict["blur"], aug_dict["blur"]),
            sigma=(0.1, float(aug_dict["blur"]))
        )

        x = blur_tf(x)
    x = add_gaussian_noise(x, aug_dict["noise_mean"], aug_dict["noise_std"])
    return x

def augment_image(
    x: torch.Tensor
) -> (torch.Tensor, dict):

    flip = bool(random.randint(0, 1))
    rot = random.randint(0, 3)

    aug_geometric_dict = {
        "flip": flip,
        "rot": rot
    }

    aug_photometric_dict = {
        "jitter": 0.3,
        "blur": 3,
        "noise_mean": 0.0,
        "noise_std": 0.1
    }

    x_aug = apply_geometric(x, aug_geometric_dict, flip_axis = 2, rot_axis0 = 1, rot_axis1 = 2)
    x_aug = apply_photometric(x_aug, aug_photometric_dict)

    return x_aug, aug_geometric_dict

def augment_mask(
    x: torch.tensor,
    aug_dict: dict
):
    x_aug = apply_geometric(x, aug_dict, flip_axis=1, rot_axis0=0, rot_axis1=1)

    return x_aug

def reverse_augmentations(
    x: torch.Tensor,
    aug_dict: dict,
    flip_axis: int = 1,
    rot_axis0: int = 0,
    rot_axis1: int = 1
):
    # 1) undo rotation
    k = aug_dict["rot"]
    if k:
        inv_k = (4 - k) % 4
        x = torch.rot90(x, k=inv_k, dims=(rot_axis0, rot_axis1))

    # 2) undo flip
    if aug_dict["flip"]:
        x = torch.flip(x, dims=[flip_axis])
    return x