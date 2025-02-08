import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

import torch
import os
import sys
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader
import json
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
import os


from Model.model import CombinedVideoModel
from datasets.dataset import Custom_Dataset_1


from Externals.utils import Return_root_dir
from Externals.logger import setup_logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Log.txt")
train_logger = setup_logger('Training_logger', train_log_path)
from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor()
   
])

dataset_root = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/datasets/custom_dataset"
train_list_path = os.path.join(dataset_root, 'sep_trainlist.txt')

train_dataset = Custom_Dataset_1(root_dir=dataset_root, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)


model = CombinedVideoModel(in_channels=3, out_channels=3).to(device)


criterion = nn.L1Loss()


with open('/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/deepspeed/deepspeed_config.json') as f:
    ds_config = json.load(f)

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config
)
print("DeepSpeed model engine initialized successfully.")
device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
torch.cuda.empty_cache()
from torch.amp import autocast  



def Train():
    model_engine.train()
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}] started")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            print(f"Batch {batch_idx} loaded, input shape: {inputs.shape}")

            model_engine.zero_grad()

            with autocast("cuda", dtype=torch.float16, enabled=(device.type == 'cuda')):
                esgran_output, unet_output = model_engine(inputs)
                print(f"esgran_output shape: {esgran_output.shape}, unet_output shape: {unet_output.shape}")
                
                loss = criterion(unet_output, esgran_output)
                print(f"Loss: {loss.item()}")

            model_engine.backward(loss)
            model_engine.step() 
            
            epoch_loss += loss.item()
            print(f"Cumulative epoch loss: {epoch_loss}")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_epoch_loss:.4f}")

    torch.cuda.empty_cache()
    model_engine.save_checkpoint("esgran_unet_checkpoint")

if __name__ == "__main__":
    Train()
