import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import torch
import os
import sys
import deepspeed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import Vimeo90KSeptuplets
import json
import torch.nn.functional as F
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.insert(0, project_root)
from model import CombinedVideoModel
from utils import Return_root_dir
from logger import setup_logger
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir,"Log.txt")
train_logger = setup_logger('Training_logger', train_log_path)


with open('deepspeed_config.json') as f:
    ds_config = json.load(f)


def Train():
