import torch
from torch.utils.data import DataLoader

from dataset import BaseballVideoDataset
from train import BaseballPitchModel


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BaseballVideoDataset("data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = BaseballPitchModel().to(device)
    model.load_state_dict(torch.load("best_pitch_model.pt", map_location=device))

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Accuracy: {correct / total:.4f}")


if __name__ == "__main__":
    main()