import os
import random
import json
import pickle
import logging
from pathlib import Path
import sys
sys.path.append('..')
import numpy as np
import torch
import torch.nn as nn


logger = logging.getLogger()

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if isinstance(log_file, Path):
        log_file = str(log_file)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    # [2020-08-07 15:07:34,551 - prepare_fold_data.py:26 - root]: Start making K-Fold training data.
    fmt = "[%(asctime)s - %(filename)s:%(lineno)s - %(name)s]: %(message)s"
    log_format = logging.Formatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file !='':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

def save_pickle(data, file_path):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(file_path):
    with open(str(file_path), 'rb') as f:
        data = pickle.load(f)
    return data

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def load_json(file_path):
    if not isinstance(file_path, Path):
        file_path=Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def prepare_device(n_gpu_use):
    if not n_gpu_use:
        device_type = 'cpu'
    else:
        n_gpu_use = n_gpu_use.split(',')
        device_type = f'cuda:{n_gpu_use[0]}'

    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        logger.warning('Warning: There is no GPU available on this machine, training will be performed on CPU.')
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are available on this machine."
        logger.warning(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids

def model_device(n_gpu, model):
    device, device_ids = prepare_device(n_gpu)
    device_ids = list(map(int,device_ids))
    if len(device_ids) > 1:
        logger.info(f'current {len(device_ids)} GPUs')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count