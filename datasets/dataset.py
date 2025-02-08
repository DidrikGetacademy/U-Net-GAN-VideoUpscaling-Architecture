import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class Custom_Dataset_1(Dataset):
    def __init__(self, root_dir, transform=None, 
                 input_folder='Input', target_folder='Target', 
                 num_frames=7):

        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, input_folder)
        self.target_dir = os.path.join(root_dir, target_folder)
        self.transform = transform
        self.num_frames = num_frames

      
        self.input_sequence = sorted(os.listdir(self.input_dir))
        self.target_sequence = sorted(os.listdir(self.target_dir))
        

        assert len(self.input_sequence) == len(self.target_sequence), (
            "Mismatch between number of input and target clips."
        )

    def __len__(self):
        return len(self.input_sequence)

    def load_frames(self, sequence_path):

        frame_names = sorted([
            f for f in os.listdir(sequence_path) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])

    
        if len(frame_names) >= self.num_frames:
            selected_frames = frame_names[:self.num_frames]
        else:
   
            selected_frames = frame_names.copy()
            while len(selected_frames) < self.num_frames:
                selected_frames.append(frame_names[-1])

        frames = []
        for frame_name in selected_frames:
            frame_path = os.path.join(sequence_path, frame_name)
  
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            frames.append(image)
        

        frames_tensor = torch.stack(frames, dim=1)
        return frames_tensor

    def __getitem__(self, index):

        input_clip_folder = os.path.join(self.input_dir, self.input_sequence[index])
        target_clip_folder = os.path.join(self.target_dir, self.target_sequence[index])

        input_frames = self.load_frames(input_clip_folder)
        target_frames = self.load_frames(target_clip_folder)

        return input_frames, target_frames


if __name__ == "__main__":

    root_dir = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/datasets/custom_dataset"


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

 
    dataset = Custom_Dataset_1(
        root_dir=root_dir, 
        transform=transform, 
        input_folder='Input', 
        target_folder='Target', 
        num_frames=7
    )

   
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

    # Iterate through the dataset
    for input_frames, target_frames in dataloader:
        print("Input frames shape:", input_frames.shape)   
        print("Target frames shape:", target_frames.shape)   
        break
