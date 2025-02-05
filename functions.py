import torch
from utils import Return_root_dir
from logger import setup_logger
root_dir = Return_root_dir()
train_log_path = os.path.join(root_dir,"Log.txt")
train_logger = setup_logger('Training_logger', train_log_path)



