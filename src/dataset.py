import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# racine où tu as mis tes spectrogrammes
DATA_DIR = "./data/spectrograms"


from torchvision import transforms

def get_transforms():
    """
    Transformations appliquées aux images au chargement.
    On force une taille fixe (128, 256) pour que toutes les images
    d'un batch aient la même dimension.
    """
    return transforms.Compose([
        transforms.Resize((128, 256)),              # (H, W) => 128 mel-bins x 256 "pas de temps"
        transforms.ToTensor(),                      # -> tensor [1, 128, 256] en [0,1]
    ])


def get_datasets(data_dir: str = DATA_DIR):
    """
    Retourne les Dataset (train, val, test) basés sur ImageFolder.
    data_dir/train/<genre>/*.png
    data_dir/val/<genre>/*.png
    data_dir/test/<genre>/*.png
    """
    tfm = get_transforms()

    train_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "train"),
        transform=tfm,
    )

    val_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "val"),
        transform=tfm,
    )

    test_ds = datasets.ImageFolder(
        root=os.path.join(data_dir, "test"),
        transform=tfm,
    )

    return train_ds, val_ds, test_ds


def get_dataloaders(
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle_train: bool = True,
    data_dir: str = DATA_DIR,
):
    """
    Construit les DataLoader pour train/val/test.
    """
    train_ds, val_ds, test_ds = get_datasets(data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

    batch = next(iter(train_loader))
    images, labels = batch

    print("Taille batch d'images :", images.shape)   # attendu: [8, 1, H, W]
    print("Taille batch de labels :", labels.shape)
    print("Labels dans ce batch :", labels)

    # Pour voir la correspondance index -> nom de classe
    train_ds, _, _ = get_datasets()
    print("classes :", train_ds.classes)            # ['Electronic', 'Experimental', ...]
    print("class_to_idx :", train_ds.class_to_idx)  # {'Electronic': 0, ...}
