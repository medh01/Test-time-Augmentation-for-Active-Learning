import torch
from torch.utils.data import Dataset, DataLoader
import os
import random
from PIL import Image, ImageOps
import numpy as np
from augmentation_utils import augment_image, augment_mask

Classes = {
    (51, 221, 255): 0,  # ICM
    (250, 50, 83): 1,   # TE
    (61, 245, 61): 2,   # ZP
    (255, 245, 61): 3,  # BL
    (0, 0, 0): 4,       # background
}

TARGET_SIZE = (256, 256)  # (W, H)


# Label Encoding
def mask_encoding(arr):
    h, w, _ = arr.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, idx in Classes.items():
        class_mask[np.all(arr == rgb, axis=-1)] = idx
    return class_mask


class BlastocystDataset(Dataset):
    def __init__(self, image_dir, mask_dir, seed=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = sorted(os.listdir(image_dir))
        self.seed = seed
        self.augment = augment

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        core = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, core + ".png")


        img = Image.open(img_path).convert("RGB")
        img = ImageOps.pad(img, TARGET_SIZE, method=Image.BILINEAR, color=(0, 0, 0))
        img = np.array(img, np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        mask_rgb = Image.open(mask_path).convert("RGB")
        mask_rgb = ImageOps.pad(mask_rgb, TARGET_SIZE, method=Image.NEAREST, color=(0, 0, 0))
        mask_arr = np.array(mask_rgb)
        mask = mask_encoding(mask_arr)  # [H,W] uint8
        mask = torch.from_numpy(mask).long()

        if self.augment:
            img, aug_dict = augment_image(img)
            mask = augment_mask(mask, aug_dict)
            print(aug_dict)

        return img, mask, img_name

class UnlabeledBlastocystDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_filenames = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.image_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.pad(img, TARGET_SIZE, method=Image.BILINEAR, color=(0, 0, 0))
        img = np.array(img, np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        return img, img_name




def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loaders_active(
        labeled_img_dir,
        labeled_mask_dir,
        unlabeled_img_dir,
        test_img_dir,
        test_mask_dir,
        batch_size,
        seed=None,
        augment=False,
        generator=None,
        num_workers=4,
        pin_memory=True,
):
    # 1. Labeled Dataset (with masks)
    labeled_ds = BlastocystDataset(
        image_dir=labeled_img_dir,
        mask_dir=labeled_mask_dir,
        seed=seed,
        augment=augment
    )

    # 2. Unlabeled Dataset (images only)
    unlabeled_ds = UnlabeledBlastocystDataset(
        image_dir=unlabeled_img_dir
    )

    # 3. Test Dataset (with masks)
    test_ds = BlastocystDataset(
        image_dir=test_img_dir,
        mask_dir=test_mask_dir,
        seed=seed,
        augment=False
    )

    # Create loaders
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Helps with batch normalization
    )

    unlabeled_loader = DataLoader(
        unlabeled_ds,
        batch_size=batch_size,
        shuffle=False,  # Important for sample tracking
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return labeled_loader, unlabeled_loader, test_loader