import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__()

        # Bloc 1 : Conv -> ReLU -> MaxPool -> Dropout
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (128, 256) -> (64, 128)
            nn.Dropout2d(0.25),
        )

        # Bloc 2
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (64, 128) -> (32, 64)
            nn.Dropout2d(0.25),
        )

        # Bloc 3
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (32, 64) -> (16, 32)
            nn.Dropout2d(0.25),
        )

        # feature map = [B, 64, 16, 32]
        self.flatten_dim = 64 * 16 * 32

        # Fully connected (logits ; softmax sera fait implicitement par CrossEntropyLoss !!)
        self.fc = nn.Linear(self.flatten_dim, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)           # [B, 64, 16, 32]
        x = x.view(x.size(0), -1)    # [B, 64*16*32]
        logits = self.fc(x)          # [B, num_classes]
        return logits                # PAS de softmax ici


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # paramètres
    batch_size = 4
    num_classes = 8
    in_channels = 3
    H, W = 128, 256  # comme dans ton dataset (Resize)

    # 1) instancier le modèle
    model = BaselineCNN(num_classes=num_classes, in_channels=in_channels)
    print(model)

    # 2) créer un batch factice
    x = torch.randn(batch_size, in_channels, H, W)
    logits = model(x)

    print("\n=== TEST FORWARD ===")
    print("Input shape  :", x.shape)        # attendu: [4, 3, 128, 256]
    print("Output shape :", logits.shape)   # attendu: [4, 8]

    # 3) tester la loss et le backward
    criterion = nn.CrossEntropyLoss()

    # labels factices entre 0 et num_classes-1
    y = torch.randint(low=0, high=num_classes, size=(batch_size,))

    loss = criterion(logits, y)
    print("\n=== TEST LOSS ===")
    print("Labels        :", y)
    print("Loss value    :", loss.item())

    loss.backward()
    print("\nBackward OK (aucune erreur)")
