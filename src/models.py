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

class ImprovedCRNN(nn.Module):
    """
    CRNN : CNN + GRU bidirectionnel sur l'axe temporel.
    Entrée : [B, 3, 128, 256] (spectrogrammes redimensionnés)
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 3,
                 rnn_hidden_size: int = 64):
        super().__init__()

        # Même backbone CNN que la baseline (pour comparaison honnête)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (128, 256) -> (64, 128)
            nn.Dropout2d(0.25),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (64, 128) -> (32, 64)
            nn.Dropout2d(0.25),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (32, 64) -> (16, 32)
            nn.Dropout2d(0.25),
        )

        # Après les 3 blocs :
        # x : [B, 64, 16, 32]
        #  - 16 = dimension fréquence (mel)
        #  - 32 = dimension temps
        # On va faire :
        #   - moyenne sur la dimension fréquence -> [B, 64, 32]
        #   - permute en [B, 32, 64] pour la GRU (time major)

        self.rnn_input_size = 64
        self.rnn_hidden_size = rnn_hidden_size

        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=1,
            batch_first=True,     # input [B, T, F]
            bidirectional=True,
        )

        # sortie GRU bidirectionnelle : hidden dim = 2 * hidden_size
        self.fc = nn.Linear(2 * self.rnn_hidden_size, num_classes)

    def forward(self, x):
        # CNN feature extractor
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)              # [B, 64, 16, 32]

        # moyenne sur la dimension fréquence (H=16)
        x = x.mean(dim=2)               # [B, 64, 32]

        # réorganisation pour GRU : [B, T, F]
        x = x.permute(0, 2, 1)          # [B, 32, 64] : T=32 pas de temps, F=64 features

        # GRU bidirectionnelle
        rnn_out, _ = self.gru(x)        # [B, 32, 2*hidden_size]

        # on agrège sur le temps (moyenne des pas de temps)
        x = rnn_out.mean(dim=1)         # [B, 2*hidden_size]

        logits = self.fc(x)             # [B, num_classes]
        return logits

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
