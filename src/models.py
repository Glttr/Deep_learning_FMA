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
                 rnn_hidden_size: int = 128):
        super().__init__()

        # Même backbone CNN que la baseline (pour comparaison honnête)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (128, 256) -> (64, 128)
            nn.Dropout2d(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (64, 128) -> (32, 64)
            nn.Dropout2d(0.1),
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
    
class ImprovedCRNNv2(nn.Module):
    """
    CRNN amélioré :
    - CNN plus profond avec BatchNorm
    - Pas de moyenne en fréquence : on garde C * F pour chaque pas de temps
    - GRU bidirectionnelle qui reçoit une représentation riche [B, T, C*F]
    - Classification à partir du dernier état caché (fw + bw)
    Entrée : [B, 3, 128, 256]
    """

    def __init__(self, num_classes: int = 8, in_channels: int = 3,
                 hidden_size: int = 128, n_layers: int = 1):
        super().__init__()

        # Backbone CNN plus expressif (4 blocs) avec BatchNorm
        self.cnn = nn.Sequential(
            # Bloc 1 : 3 -> 32
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (128, 256) -> (64, 128)
            nn.Dropout2d(0.2),

            # Bloc 2 : 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (64, 128) -> (32, 64)
            nn.Dropout2d(0.2),

            # Bloc 3 : 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (32, 64) -> (16, 32)
            nn.Dropout2d(0.3),

            # Bloc 4 : 128 -> 128 (on garde la même largeur de canaux)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (16, 32) -> (8, 16)
            nn.Dropout2d(0.3),
        )

        # Après le CNN, si l'entrée est [B, 3, 128, 256] :
        #   H (freq) : 128 -> 64 -> 32 -> 16 -> 8
        #   W (temps): 256 -> 128 -> 64 -> 32 -> 16
        # x : [B, 128, 8, 16]
        self.freq_out = 8
        self.time_out = 16
        self.cnn_channels = 128

        # Chaque pas de temps voit un vecteur de taille C * F
        self.rnn_input_size = self.cnn_channels * self.freq_out  # 128 * 8 = 1024

        self.gru = nn.GRU(
            input_size=self.rnn_input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,     # input [B, T, F]
            bidirectional=True,
        )

        # On concatène les états finaux fw+bw
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        # x : [B, 3, 128, 256]
        x = self.cnn(x)                  # [B, 128, 8, 16]
        B, C, F, T = x.shape

        # On veut une séquence le long de l'axe temps (T = 16)
        x = x.permute(0, 3, 1, 2)        # [B, T, C, F]
        x = x.reshape(B, T, C * F)       # [B, T, C*F] = [B, 16, 1024]

        out, h = self.gru(x)             # out: [B, T, 2H]; h: [2*n_layers, B, H]

        # Dernier layer : h[-2] (fw), h[-1] (bw)
        h_fw = h[-2]                     # [B, H]
        h_bw = h[-1]                     # [B, H]
        h_cat = torch.cat([h_fw, h_bw], dim=1)  # [B, 2H]

        logits = self.fc(h_cat)          # [B, num_classes]
        return logits


class CNN_other(nn.Module):
    def __init__(self, num_classes: int = 8, in_channels: int = 3):
        super().__init__()

        # Bloc 1 : test avec plus large que la baseline
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),  # (128,256) -> (128,256)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (128,256) -> (64,128)
            nn.Dropout2d(0.25),
        )

        # Bloc 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),           # (64,128) -> (64,128)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (64,128) -> (32,64)
            nn.Dropout2d(0.25),
        )

        # Bloc 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # (32,64) -> (32,64)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (32,64) -> (16,32)
            nn.Dropout2d(0.25),
        )

        # À ce stade : feature map = [B, 128, 16, 32]
        self.flatten_dim = 128 * 16 * 32  # = 65536

        # Fully connected plus riche que le baseline
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(p=0.5)

        self.out = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)           # [B, 32, 64, 128]
        x = self.block2(x)           # [B, 64, 32, 64]
        x = self.block3(x)           # [B, 128, 16, 32]

        x = x.view(x.size(0), -1)    # [B, 128*16*32]

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        logits = self.out(x)         # [B, num_classes]
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
