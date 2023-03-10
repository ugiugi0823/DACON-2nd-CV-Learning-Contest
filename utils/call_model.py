# call_model.py
import os
import csv
import cv2
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold
import random
from time import time
import IPython
import copy

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from warmup_scheduler import GradualWarmupScheduler

# from src.train import train_model
# from utils.imageprocess import image_transformer, image_processor
# from utils.EarlyStopping import EarlyStopping
# from utils.dataloader import CustomDataLoader
# from utils.radams import RAdam
# from utils.call_model import CallModel
from torchvision.models import resnet18
from efficientnet_pytorch import EfficientNet

from src.model import *
from tqdm import tqdm
import logging

class CallModel():
    def __init__(self, model_type=None, pretrained=True, logger=None, path='/content/drive/MyDrive/pretraines_model'):
        
        # MODEL TYPE
        if model_type == 'resnet18':
            base_model = Resnet18()
            weight_path = os.path.join(path, 'resnet18.pth')

        elif model_type == 'efficientnetb1':
            base_model = EfficientnetB1()
            weight_path = os.path.join(path, 'adv-efficientnet-b1.pth')
            
        elif model_type == 'efficientnetb0':
            base_model = EfficientnetB0()
            weight_path = os.path.join(path, 'adv-efficientnet-b0.pth')
            
        elif model_type == 'efficientnetb2':
            base_model = EfficientnetB2()
            weight_path = os.path.join(path, 'adv-efficientnet-b2.pth')
            
        else:
            raise Exception(f"No such model type: {model_type}")
        
        
        # LOAD PRETRAINED WEIGHTS
        if pretrained:
            logger.info(f"Using pretrained model. Loading weights from {weight_path}")
            base_model = CallModel._load_weights(base_model, weight_path)
            
            # b5 model
            nn.init.xavier_normal_(base_model.block[0]._fc.weight)
           
        else:
            logger.info(f"Not using pretrained model.")
        
        self.model = base_model
            
    def model_return(self):
        return self.model
        
    @staticmethod
    def _load_weights(model, path):
        model.load_state_dict(torch.load(path))
        return model
