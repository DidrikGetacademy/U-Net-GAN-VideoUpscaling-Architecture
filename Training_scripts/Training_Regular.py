import torch
import os
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
from Model.model import CombinedVideoModel
from datasets.dataset import Vimeo90KSeptuplets
from torch.amp import autocast, GradScaler  # Prefer torch.cuda.amp for GPU

# Setup paths and logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from Externals.utils import Return_root_dir
from Externals.logger import setup_logger


root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Log.txt")
train_logger = setup_logger('Training_logger', train_log_path)


# Dataset
dataset_root = r"C:\Users\didri\Desktop\Programmering\ArtificalintelligenceModels\UNet_Gan_model_Video_Enchancer\vimeo_septuplet"
train_list_path = os.path.join(dataset_root, 'sep_trainlist.txt')
train_dataset = Vimeo90KSeptuplets(root_dir=dataset_root, split_file=train_list_path)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
device = torch.device("cuda:0") 
model = CombinedVideoModel(in_channels=3, out_channels=3).to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = GradScaler()  
model = nn.DataParallel(
    model,
    device_ids=[0, 1]  
)
num_epochs = 10

def Train():
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch: [{epoch+1} started]")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            print(type(batch))
            print(f"Batch {batch_idx}")

     
     
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(f"Accumulated epoch loss: [{epoch_loss}]")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch: [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    torch.cuda.empty_cache()
    torch.save(model.state_dict(), "esgran_unet_checkpoint.pth")

if __name__ == "__main__":
    Train()
