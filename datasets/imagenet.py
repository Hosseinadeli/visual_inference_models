#!/usr/bin/env python

import collections
import os

import numpy as np
#import PIL.Image
from PIL import Image, ImageOps 
import scipy.io
import torch
from torch.utils import data
import torchvision
from tqdm import tqdm
import pandas as pd
import csv

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


def fetch_ImageNet(args, split='train', transform=None,
                 horizontal_flip=False, upper=None):
    
    data_path = '/share/data/imagenet-pytorch'

    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'train'),
        torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))
    
    data_loader_train = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    

    dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))
    
    data_loader_val = torch.utils.data.DataLoader(dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    
    return data_loader_train, data_loader_val
        
