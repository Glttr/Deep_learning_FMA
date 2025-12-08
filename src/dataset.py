import os
import sys
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# racine il y a les spectrogrammes
DATA_DIR = "./data/spectrograms"


from torchvision import transforms


def generate_clean_sets():
    if not os.path.isdir("./data/raw/fma_metadata"):
        print('fma_metadata is not downloaded or well located ! Download fma_metadata in data/raw. Do the same for fma_small as well !')
        return None

    cols = [
        ('track', 'genre_top'),
        ('track', 'genres'),
        ('track', 'genres_all')
    ]
    tracks = pd.read_csv('./data/raw/fma_metadata/tracks.csv', index_col=0, header=[0, 1]) # index_col=0, header=[0, 1, 2]
    small = tracks[tracks['set', 'subset'] == 'small']
    small_train = small[small['set', 'split'] == 'training']
    small_val = small[small['set', 'split'] == 'validation']
    small_test = small[small['set', 'split'] == 'test']


    # train
    train_genres = small_train[cols].copy()
    train_genres.columns = train_genres.columns.get_level_values(-1)

    # val
    val_genres = small_val[cols].copy()
    val_genres.columns = val_genres.columns.get_level_values(-1)

    # test
    test_genres = small_test[cols].copy()
    test_genres.columns = test_genres.columns.get_level_values(-1)

    train_genres.head()

    train_genres.to_csv('./data/raw/fma_metadata/small_train_genres.csv', index='track_id')
    val_genres.to_csv('./data/raw/fma_metadata/small_val_genres.csv', index='track_id')
    test_genres.to_csv('./data/raw/fma_metadata/small_test_genres.csv', index='track_id')
    print('./data/raw/fma_metadata/small_test_genres.csv, small_train_genres.csv and small_val_genres.csv have been generated !')
    return None




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

    if not os.path.isfile("./data/raw/fma_metadata/small_train_genres.csv"):
        print("Clean genre CSVs not found, generating them with generate_clean_sets()...")
        generate_clean_sets()

    required_dirs = [
        "./data/spectrograms/train",
        "./data/spectrograms/val",
        "./data/spectrograms/test"
    ]

    missing = [d for d in required_dirs if not os.path.isdir(d)]

    if missing:
        print("Some spectrogram folders are missing:")
        for d in missing:
            print(" -", d)

        # On crée l'arborescence vide au cas où
        for d in required_dirs:
            os.makedirs(d, exist_ok=True)

        print(
            "\nFolders have been created, but they are empty.\n"
            "Please run your script that generates mel-spectrogram PNGs "
            "into data/spectrograms/train|val|test before training."
        )   
        sys.exit(0)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8)

    batch = next(iter(train_loader))
    images, labels = batch

    print("Taille batch d'images :", images.shape)   # attendu: [8, 3, H, W]
    print("Taille batch de labels :", labels.shape)
    print("Labels dans ce batch :", labels)

    # Pour voir la correspondance index -> nom de classe
    train_ds, _, _ = get_datasets()
    print("classes :", train_ds.classes)            # ['Electronic', 'Experimental', ...]
    print("class_to_idx :", train_ds.class_to_idx)  # {'Electronic': 0, ...}
