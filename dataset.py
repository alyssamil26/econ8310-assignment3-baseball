import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class BaseballVideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, image_size=(112, 112)):
        self.root_dir = Path(root_dir)
        self.clip_len = clip_len
        self.image_size = image_size

        self.classes = {"no_pitch": 0, "pitch": 1}
        self.samples = self._build_index()

    def _build_index(self):
        samples = []

        for class_name, label in self.classes.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for clip_dir in class_dir.iterdir():
                if clip_dir.is_dir():
                    samples.append((clip_dir, label))

        return samples

    def __len__(self):
        return len(self.samples)

    def _load_frame(self, path):
        img = Image.open(path).convert("RGB")
        img = img.resize(self.image_size)

        img = np.array(img) / 255.0
        img = torch.tensor(img).float().permute(2, 0, 1)

        return img

    def __getitem__(self, idx):
        clip_dir, label = self.samples[idx]

        frame_files = sorted(clip_dir.glob("*.jpg"))

        if len(frame_files) >= self.clip_len:
            frame_files = frame_files[:self.clip_len]
        else:
            frame_files += [frame_files[-1]] * (self.clip_len - len(frame_files))

        frames = [self._load_frame(f) for f in frame_files]

        video = torch.stack(frames).permute(1, 0, 2, 3)

        return video, torch.tensor(label)


if __name__ == "__main__":
    dataset = BaseballVideoDataset("data/train")

    print(f"Total samples: {len(dataset)}")

    video, label = dataset[0]

    print("Video shape:", video.shape)
    print("Label:", label)