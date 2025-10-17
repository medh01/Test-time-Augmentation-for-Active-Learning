import os
import shutil
import random
import torch
from tqdm import tqdm

def create_active_learning_pools(
        BASE_DIR,
        label_split_ratio=0.1,
        test_split_ratio=0.2,
        shuffle=True
):
    """Creates directories and splits data into labeled, unlabeled, and test pools.

    This function sets up the directory structure for an active learning experiment.
    It splits the available images into training (labeled and unlabeled) and test sets
    based on the provided ratios, and copies the corresponding images and masks into
    the appropriate directories.

    Args:
        BASE_DIR (str): The base directory containing the 'images' and 'masks' folders.
        label_split_ratio (float, optional): The ratio of the data to be used as the initial labeled pool. Defaults to 0.1.
        test_split_ratio (float, optional): The ratio of the data to be used for the test set. Defaults to 0.2.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.

    Returns:
        dict: A dictionary containing the paths to the created directories.
    """
    # Create directories
    dirs = {
        'labeled_img': os.path.join(BASE_DIR, "Labeled_pool", 'labeled_images'),
        'labeled_mask': os.path.join(BASE_DIR, "Labeled_pool", 'labeled_masks'),
        'unlabeled_img': os.path.join(BASE_DIR, "Unlabeled_pool", 'unlabeled_images'),
        'unlabeled_mask': os.path.join(BASE_DIR, "Unlabeled_pool", 'unlabeled_masks'),
        'test_img': os.path.join(BASE_DIR, "Test", 'test_images'),
        'test_mask': os.path.join(BASE_DIR, "Test", 'test_masks')
    }

    dirs["labeled_img_dir"] = dirs["labeled_img"]
    dirs["labeled_mask_dir"] = dirs["labeled_mask"]
    dirs["unlabeled_img_dir"] = dirs["unlabeled_img"]
    dirs["test_img_dir"] = dirs["test_img"]
    dirs["test_mask_dir"] = dirs["test_mask"]

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    # Get image list
    img_dir = os.path.join(BASE_DIR, 'images')
    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith("bmp")])

    if shuffle:
        random.shuffle(images)

    # Split images
    n_test = int(len(images) * test_split_ratio)
    n_labeled = int(len(images) * label_split_ratio)

    test_split = images[:n_test]
    labeled_split = images[n_test:n_test + n_labeled]
    unlabeled_split = images[n_test + n_labeled:]

    def copy_files(file_list, img_dest, mask_dest):

        for im in file_list:
            base_name = os.path.splitext(im)[0]

            # Copy image
            src_img = os.path.join(img_dir, im)
            dst_img = os.path.join(img_dest, im)
            shutil.copy(src_img, dst_img)

            # Copy mask
            mask_file = f"{base_name}.png"
            src_mask = os.path.join(BASE_DIR, 'masks', mask_file)
            dst_mask = os.path.join(mask_dest, mask_file)

            if os.path.exists(src_mask):
                shutil.copy(src_mask, dst_mask)
            else:
                print(f"Warning: Mask not found for {im} - {src_mask}")

    copy_files(test_split, dirs['test_img'], dirs['test_mask'])
    copy_files(labeled_split, dirs['labeled_img'], dirs['labeled_mask'])
    copy_files(unlabeled_split, dirs['unlabeled_img'], dirs['unlabeled_mask'])

    return dirs

def reset_data(base_dir):
    """Removes the directories created by `create_active_learning_pools`.

    This function is useful for cleaning up the directory structure and starting a new
    experiment from scratch.

    Args:
        base_dir (str): The base directory where the data pools were created.
    """
    # Directories to remove
    dirs_to_remove = [
        os.path.join(base_dir, "Labeled_pool"),
        os.path.join(base_dir, "Unlabeled_pool"),
        os.path.join(base_dir, "Test")
    ]

    for dir_path in dirs_to_remove:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)


def move_images_with_dict(
        base_dir: str,
        labeled_dir: str,
        unlabeled_dir: str,
        score_dict: dict,
        num_to_move: int = 2
):
    """Moves a specified number of the most uncertain images from the unlabeled to the labeled pool.

    This function takes a dictionary of image scores (uncertainties), sorts them in
    descending order, and moves the top `num_to_move` images (and their corresponding
    masks, if they exist) from the unlabeled pool to the labeled pool.

    Args:
        base_dir (str): The base directory for the data pools.
        labeled_dir (str): The name of the labeled pool directory.
        unlabeled_dir (str): The name of the unlabeled pool directory.
        score_dict (dict): A dictionary where keys are image filenames and values are their uncertainty scores.
        num_to_move (int, optional): The number of images to move. Defaults to 2.
    """
    # Sort by descending uncertainty (most uncertain first)
    sorted_items = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    moved = 0
    for im, score in sorted_items:
        if moved >= num_to_move:
            break

        # Clean filename and get base name
        im_clean = im.strip()
        base_name = os.path.splitext(im_clean)[0]

        # Image paths
        src_im = os.path.join(base_dir, unlabeled_dir, "unlabeled_images", im_clean)
        dst_im = os.path.join(base_dir, labeled_dir, "labeled_images", im_clean)

        # Mask paths
        mask_name = base_name + ".png"
        src_msk = os.path.join(base_dir, unlabeled_dir, "unlabeled_masks", mask_name)
        dst_msk = os.path.join(base_dir, labeled_dir, "labeled_masks", mask_name)

        # Verify image exists
        if not os.path.exists(src_im):
            print(f"[WARN] Image not found: {src_im}")
            continue

        # Move image
        shutil.copy(src_im, dst_im)
        os.remove(src_im)
        print(f"[MOVE] IMAGE {im_clean} (Uncertainty: {score:.4f})")

        # Move mask if exists
        if os.path.exists(src_msk):
            shutil.copy(src_msk, dst_msk)
            os.remove(src_msk)
            print(f"[MOVE]  MASK {mask_name}")
        else:
            print(f"[WARN] Mask not found: {src_msk}")

        moved += 1

    print(f"Moved {moved} most uncertain images from {unlabeled_dir} → {labeled_dir}.")

def score_unlabeled_pool(unlabeled_loader, model, score_fn, T=8, num_classes=5, device="cuda"):
    """Scores the unlabeled pool of images using a given scoring function.

    This function iterates through the unlabeled data, applies the specified scoring function
    (e.g., an acquisition function like BALD or max entropy) to each image, and returns a
    dictionary of image filenames and their corresponding scores.

    Args:
        unlabeled_loader (DataLoader): The data loader for the unlabeled pool.
        model (nn.Module): The model to use for scoring.
        score_fn (callable): The function to use for calculating the uncertainty scores.
        T (int, optional): The number of Monte Carlo samples for stochastic forward passes. Defaults to 8.
        num_classes (int, optional): The number of classes. Defaults to 5.
        device (str, optional): The device to use for computation. Defaults to "cuda".

    Returns:
        dict: A dictionary where keys are image filenames and values are their uncertainty scores.
    """
    model.to(device).train()
    scores, fnames = [], []
    with torch.no_grad():
        for imgs, names in tqdm(unlabeled_loader, desc="Scoring", leave=False):
            imgs = imgs.to(device)
            s = score_fn(model, imgs, T=T, num_classes=num_classes)
            scores.extend(s.cpu().tolist())
            fnames.extend(names)
    return dict(zip(fnames, scores))