import argparse
from sched import scheduler
import numpy as np
from torch import is_vulkan_available, nn
import torch
import random
from tqdm import tqdm as tqdm
from os.path import join, exists
import matplotlib.pyplot as plt
import os
from torchtoolbox.tools import mixup_data, mixup_criterion
from torch.utils.tensorboard import SummaryWriter

from model import ResNet18_For_CIFAR_100, resnet18
from data_loader import get_cifar_loader
from utils import rand_bbox





if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='hw04')
    parser.add_argument('--model', 
                    type=str,
                    default='sb')
    args = parser.parse_args()    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'               
    
    train_loader, val_loader, test_loader = get_cifar_loader(batch_size=128, n_items=-1, data_aug=False, cutout=False)
    print(len(train_loader.dataset), len(test_loader.dataset), len(val_loader.dataset))
      
    model = resnet18()    
    
    cpt = torch.load(join('model', args.model))
    
    model.load_state_dict(cpt)
    
    # Test.
    model.eval()
    test_correct = 0
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        
        pred = model(x)
        test_correct += (pred.argmax(1) == y).type(torch.float).sum().item()    
    
    test_correct /= len(test_loader.dataset)
    
    print("The test accuracy is: %f" % test_correct)    