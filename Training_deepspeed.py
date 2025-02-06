import torch.multiprocessing as mp
#mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader
import json
import gc
from model import CombinedVideoModel
from dataset import Vimeo90KSeptuplets
from torch import autocast
# Setup paths and logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from utils import Return_root_dir
from logger import setup_logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir, "Log.txt")
train_logger = setup_logger('Training_logger', train_log_path)



# Load dataset
dataset_root = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/vimeo_septuplet"
train_list_path = os.path.join(dataset_root, 'sep_trainlist.txt')

train_dataset = Vimeo90KSeptuplets(root_dir=dataset_root, split_file=train_list_path)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CombinedVideoModel(in_channels=3, out_channels=3).to(device)



criterion = nn.L1Loss()     
# Load DeepSpeed configuration
with open('deepspeed_config.json') as f:
    ds_config = json.load(f)

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config
)
print(f"DeepSpeed model engine initialized successfully.")


# Training loop

num_epochs = 10


def Train():

    model_engine.train()

    for epoch in range(num_epochs):
        print(f"Epoch: [{epoch+1} started]")
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx} loaded, batch size: {batch.size()}")
            inputs = batch.to(device)
            print(f"inputs: [{inputs.shape}]")
            model_engine.zero_grad()
            with autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                esgran_output, unet_output = model_engine(inputs)  
                print(f"esgran_output: [{esgran_output.shape}, unet_output: {unet_output.shape}]")
                loss = criterion(unet_output,esgran_output)
                print(f"loss:[{loss.item()}]")

         
            model_engine.backward(loss) 
            model_engine.step()
            model_engine.update()

            epoch_loss += loss.item()
            print(f"epochloss: [{epoch_loss}]")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch: [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

    torch.cuda.empty_cache()


    model_engine.save_checkpoint("esgran_unet_checkpoint")
if __name__ == "__main__":
    Train()
