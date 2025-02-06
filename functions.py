import torch
from utils import Return_root_dir
from logger import setup_logger
import torchvision.transforms as transforms
import os
import sys
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir,"Log.txt")
train_logger = setup_logger('Training_logger', train_log_path)
from dataset import Vimeo90KSeptuplets
from torch.utils.data import DataLoader


def test_dataset():
   dataset_root = "/mnt/c/Users/didri/Desktop/Programmering/ArtificalintelligenceModels/UNet_Gan_model_Video_Enchancer/vimeo_septuplet" 
   train_list_path = os.path.join(dataset_root, 'sep_trainlist.txt')
   test_list_path = os.path.join(dataset_root, 'sep_testlist.txt')
   transform = transforms.Compose([
        transforms.ToTensor(),

    ])

   train_dataset = Vimeo90KSeptuplets(root_dir=dataset_root, split_file=train_list_path, transform=transform)
   train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

 
   test_dataset = Vimeo90KSeptuplets(root_dir=dataset_root, split_file=test_list_path, transform=transform)
   test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)


   for batch in train_loader:
      print("Train batch shape:", batch.shape)
      break

 
   for batch in test_loader:
        print("Test batch shape:", batch.shape)
        break
