import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import sys

class Vimeo90KSeptuplets(Dataset):
    def __init__(self, root_dir, split_file=None, transform=None):
        self.sequences_dir = os.path.join(root_dir, 'sequences')
        self.transform = transform

        if split_file is not None:
            self.septuplet_list = self._load_split_list(split_file)
        else:
            self.septuplet_list = self._make_dataset()

    def _load_split_list(self, split_file):
        septuplets = []
        with open(split_file, 'r') as f:
            for line in f:
                rel_path = line.strip()
                septuplet_path = os.path.join(self.sequences_dir, rel_path)
                frames = [os.path.join(septuplet_path, f'im{i}.png') for i in range(1, 8)]
                if os.path.isdir(septuplet_path) and all(os.path.exists(frame) for frame in frames):
                    septuplets.append(septuplet_path)
                else:
                    print(f"Warning: {septuplet_path} is missing or incomplete!")
                    raise ValueError
        return septuplets

    def _make_dataset(self):
        septuplets = []
        for folder in sorted(os.listdir(self.sequences_dir)):
            folder_path = os.path.join(self.sequences_dir, folder)
            if os.path.isdir(folder_path):
                for subfolder in sorted(os.listdir(folder_path)):
                    subfolder_path = os.path.join(folder_path, subfolder)
                    if os.path.isdir(subfolder_path):
                        frames = [os.path.join(subfolder_path, f'im{i}.png') for i in range(1, 8)]
                        if all(os.path.exists(frame) for frame in frames):
                            septuplets.append(subfolder_path)
                        else:
                            print(f"Warning: missing frames in {subfolder_path}")
                            raise ValueError
        return septuplets

    def __len__(self):
        return len(self.septuplet_list)

    def __getitem__(self, idx):
        septuplet_path = self.septuplet_list[idx]
        frames = []
        for i in range(1, 8):
            img_path = os.path.join(septuplet_path, f'im{i}.png')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.ToTensor()(img)
            frames.append(img)

        # Stack frames into a video tensor with shape [C, T, H, W]
        video = torch.stack(frames, dim=1)  # Shape: [3, 7, H, W]

        # Use the original frames as the target for denoising/enhancement
        target = video.clone()  # Assuming you want the model to learn to enhance/denoise

        return video, target
