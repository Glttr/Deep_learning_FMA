import argparse
import torch
import torch.nn as nn

from .dataset import get_dataloaders
from .models import BaselineCNN, ImprovedCRNN, ImprovedCRNNv2
from .train import set_seed, get_device, evaluate  # on réutilise ta fonction evaluate


def get_args():
    parser = argparse.ArgumentParser(description="Evaluation script for CNN models")

    parser.add_argument(
        "--model",
        type=str,
        default="baseline",
        choices=["baseline", "improved", "improved_v2"],
        help="Choose which model to evaluate",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt). If not provided, use ./outputs/checkpoints/<model>_best.pt",
    )

    return parser.parse_args()


def main():
    # ==== Reproductibilité ====
    set_seed(42)
    device = get_device()
    print(f"Device utilisé : {device}")

    # ==== Arguments ====
    args = get_args()
    model_name = args.model
    batch_size = args.batch_size
    ckpt_path = args.checkpoint

    if ckpt_path is None:
        ckpt_path = f"./outputs/checkpoints/{model_name}_best.pt"

    print(f"=== Configuration évaluation ===")
    print(f"Model         : {model_name}")
    print(f"Batch size    : {batch_size}")
    print(f"Checkpoint    : {ckpt_path}")

    # ==== Modèle ==== 
    if model_name == "baseline":
        model = BaselineCNN(num_classes=8, in_channels=3).to(device)
    elif model_name == "improved":
        model = ImprovedCRNN(num_classes=8, in_channels=3).to(device)
    elif model_name == "improved_v2":
        model = ImprovedCRNNv2(num_classes=8, in_channels=3).to(device)
    else:
        raise ValueError(f"Model name inconnu : {model_name}")

    # ==== Chargement du checkpoint ====
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Checkpoint chargé depuis epoch : {checkpoint.get('epoch', 'N/A')}")
    print(f"train_acc associée               : {checkpoint.get('train_acc', 'N/A')}")
    print(f"val_acc associée               : {checkpoint.get('val_acc', 'N/A')}")

    # ==== Data (on a besoin au moins du test_loader) ====
    _, _, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,
    )

    # ==== Critère (doit être le même que pour l'entraînement) ====
    criterion = nn.CrossEntropyLoss()

    # ==== Évaluation sur le test set ====
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print("\n===== Évaluation sur le test set =====")
    print(f"Test loss : {test_loss:.4f} | Test acc : {test_acc:.4f}")


if __name__ == "__main__":
    main()
