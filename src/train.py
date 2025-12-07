import argparse
import random
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from .dataset import get_dataloaders
from .models import BaselineCNN

def get_args():
    parser = argparse.ArgumentParser(description="Training script for CNN models")

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "improved"],
        help="Choose which model to train"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    return parser.parse_args()


def set_seed(seed: int = 42):
    """Fixe toutes les sources d'aléatoire pour garantir la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Options PyTorch pour rendre les convolutions déterministes
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    num_batches = len(dataloader)

    for batch_idx, (images, labels) in enumerate(dataloader, start=1):

        # --- Forward / backward ---
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # --- Stats ---
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # 👉 AFFICHAGE DU PROGRÈS
        if batch_idx % 10 == 0 or batch_idx == num_batches:
            print(f"  Batch {batch_idx}/{num_batches}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc



def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # ==== Reproductibilité ====
    set_seed(42)
    device = get_device()
    print(f"Device utilisé : {device}")


    # ==== Arguments ====
    args = get_args()

    model_name = args.model               # "baseline" ou "improved"
    num_epochs = args.epochs              # ex: 10 ou autre
    batch_size = args.batch_size          # ex: 32
    lr = args.lr                          # ex: 1e-3

    print(f"=== Configuration ===")
    print(f"Model       : {model_name}")
    print(f"Epochs      : {num_epochs}")
    print(f"Batch size  : {batch_size}")
    print(f"Learning rate : {lr}")


    ##### Décommenter la suite, si on veut manip manuellement ######
    # # ==== hyperparamètres de base (baseline) ====
    num_classes = 8
    # batch_size = 16
    # num_epochs = 8
    # lr = 1e-3


    # # ==== Choix du modèle ====
    # model_name = "baseline"   # plus tard mettres "improved"
    #################################################################

    # ==== Modèle ==== 
    if model_name == "baseline":
        model = BaselineCNN(num_classes=num_classes, in_channels=3).to(device)
    elif model_name == "improved":
        from .models import ImprovedCNN
        model = ImprovedCNN(num_classes=num_classes, in_channels=3).to(device)
    else:
        raise ValueError(f"Model name inconnu : {model_name}")

    print(f"Modèle utilisé : {model_name}")

    # ==== Dossiers de sortie ====
    checkpoint_dir = "./outputs/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")

    if device.type == "cuda":
        gpu_index = torch.cuda.current_device()  # en général 0
        torch.cuda.set_per_process_memory_fraction(0.7, device=gpu_index)
        print("Limite de mémoire GPU : 70% de la VRAM")


    # ==== Data ====
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,   # tu peux passer à 2 ou 4 plus tard sous Windows si ça va
    )

    # Loss: Cross-entropy
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    # ==== Boucle d'entraînement ====
    for epoch in range(1, num_epochs + 1):
        print(f"\n===== Epoch {epoch}/{num_epochs} =====")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Train loss : {train_loss:.4f} | Train acc : {train_acc:.4f}")
        print(f"Val   loss : {val_loss:.4f} | Val   acc : {val_acc:.4f}")
                # === Sauvegarde du meilleur modèle (selon val_acc) ===
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            num_classes = 8
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "num_classes": num_classes,
                    "batch_size": batch_size,
                    "num_epochs": num_epochs,
                    "lr": lr,
                },
            }, best_ckpt_path)
            print(f"  >> Nouveau meilleur modèle sauvegardé : {best_ckpt_path} (val_acc={val_acc:.4f})")

    print(f"\nMeilleure val_acc atteinte : {best_val_acc:.4f}")


    # ==== Évaluation finale sur le test set ====
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\n===== Évaluation finale sur le test set =====")
    print(f"Test loss : {test_loss:.4f} | Test acc : {test_acc:.4f}")


if __name__ == "__main__":
    main()
