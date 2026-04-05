import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BaseballVideoDataset


class BaseballPitchModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BaseballVideoDataset("data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = BaseballPitchModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "best_pitch_model.pt")
    print("Model saved!")


if __name__ == "__main__":
    main()