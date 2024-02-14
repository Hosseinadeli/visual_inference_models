import os, pathlib
import csv

from typing import Callable

import torch
import torchvision
from torch.utils import data
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import PIL.Image
import albumentations as A
import cv2
from collections import Counter
import scipy.io

class BFMDataset(data.Dataset):
    def __init__(self, data_dir, img_list, labels, normalization_meta, imsize=None, dataset=None):
        self.data_dir=data_dir
        self.img_list=img_list
        self.labels=labels #[num_imgs, #nodes]
        self.dataset=dataset
        self.normalize_mean=normalization_meta['mean']
        self.normalize_std=normalization_meta['std']

        if imsize==128:
            self.resize=146
            self.imsize=128
        elif imsize==224:
            self.resize=256
            self.imsize=224

    def verify_imgfile(self):
        img_paths=[os.path.join(self.data_dir, img_fn) + '.jpg' for img_fn in self.img_list]
        for img_path in img_paths:
            assert os.path.exists(img_paths), f'image: {img_path} not found.'

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_fn=self.img_list[index]

        img_path = os.path.join(self.data_dir, img_fn) + '.jpg'
        im=PIL.Image.open(img_path)
        im=transforms.ToTensor()(im)
        im=self.crop_head(im,extend=0.1)
        im=transforms.Resize(self.resize)(im)

        if self.dataset=='training':
            im = transforms.RandomCrop(self.imsize)(im)
            im = transforms.RandomGrayscale(p=0.2)(im)
        else:
            im = transforms.CenterCrop(self.imsize)(im)

        im=transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)(im)

        img_index=int(img_fn) #image filename corresponds to labels numpy array's row
        label=torch.as_tensor(self.labels[img_index,:].copy(),dtype=torch.float32)

        return im, label

    def crop_head(self,im,extend=None):
        device=im.device
        corners=torch.stack([im[:,0,0],im[:,0,-1],im[:,-1,0],im[:,-1,-1]])
        background=torch.mode(corners,axis=0).values.to(device)
        background=torch.all(im.permute(1,2,0)==background,axis=2)

        w=torch.where(torch.all(background,axis=0)==False)[0]
        h=torch.where(torch.all(background,axis=1)==False)[0]
        x1,x2=w.min().item(),w.max().item()
        y1,y2=h.min().item(),h.max().item()

        if type(extend) is float:
            x_max,y_max=im.shape[1],im.shape[2]
            x_center=torch.true_divide(x1+x2,2)
            y_center=torch.true_divide(y1+y2,2)
            x_extend=torch.true_divide(x2-x1,2)*(1+extend)
            y_extend=torch.true_divide(y2-y1,2)*(1+extend)
            x1,x2=max(0,int(x_center-x_extend)),min(int(x_center+x_extend),x_max)
            y1,y2=max(0,int(y_center-y_extend)),min(int(y_center+y_extend),y_max)

        crop=im[:,y1:y2+1,x1:x2+1]

        return crop

def BFMInverse(train_dir, train_label_path, meta_dir, test_dir=None, test_label_path=None, imsize=None, batch_size=256, val_size=0.1, num_workers=16, subsample_dataset_fraction=None):

    assert os.path.exists(train_dir), f'path: {train_dir} not found.'
    train_dir = train_dir
    assert os.path.exists(train_label_path), f'path: {train_label_path} not found.'
    train_label_path = train_label_path

    channel_dir=os.path.join(meta_dir, 'channel_meta.npz')
    assert os.path.exists(channel_dir), f'path: {channel_dir} not found.'
    normalize_meta_npz=np.load(channel_dir)

    if test_dir is not None:
        assert os.path.exists(test_dir), f'path: {test_dir} not found.'
        test_dir = test_dir
        assert os.path.exists(test_label_path), f'path: {test_label_path} not found.'
        test_label_path=test_label_path

    imsize=imsize
    batch_size=batch_size
    val_size=val_size
    num_workers=num_workers
    subsample_dataset_fraction=subsample_dataset_fraction

    stage = 'fit'
    if stage == 'fit' or stage is None:
        img_indices=sorted(os.listdir(train_dir))
        img_indices=[img_index.split('.')[0] for img_index in img_indices]

        print('Splitting train/validation data using random state',42)
        X_train, X_val, y_train, y_val = train_test_split(img_indices, img_indices, test_size=val_size, random_state=42)

        train_list = X_train
        val_list = X_val

        if subsample_dataset_fraction is not None:
            train_list = train_list[:int(len(train_list)*subsample_dataset_fraction)]
            val_list = val_list[:int(len(val_list)*subsample_dataset_fraction)]

        train_labels=np.load(train_label_path, mmap_mode='r')

    if stage == 'testing':
        img_indices=sorted(os.listdir(test_dir))
        img_indices=[img_index.split('.')[0] for img_index in img_indices]

        latents=np.load(test_label_path)
        keys=latents.files
        test_labels=np.hstack(tuple([latents[key] for key in keys]))


    train_data = BFMDataset(train_dir, train_list, train_labels, normalize_meta_npz, imsize, 'training')
    val_data = BFMDataset(train_dir, val_list, train_labels, normalize_meta_npz, imsize, 'validation')

    return train_data, val_data

# def test_dataloader(self):
#     raise NotImplementedError



class BFMIdentityDataset(data.Dataset):
    def __init__(self, data_dir, im_list, label_list, imsize=224, dataset='training'):
        self.data_dir = data_dir
        self.im_list = im_list
        self.label_list = label_list
        self.dataset = dataset

        if imsize == 224:
            self.imsize=224
            self.resize=256
        elif imsize == 128:
            self.imsize=128
            self.resize=146

        self.normalization_func=transforms.Normalize(mean=[0.55537174, 0.50970546, 0.48330758],std=[0.28882495, 0.26824081, 0.26588868])

        self.im_info=pd.DataFrame({'class_id':self.label_list,'image_path':self.im_list})

    def verify_imgfile(self):
        for im_path in self.im_info['image_path']:
            assert os.path.exists(im_path), f'image: {im_path} not found.'

    def __len__(self):
        return len(self.im_info)

    def __getitem__(self, index):
        info=self.im_info.iloc[index]
        im_path=info.image_path
        label=info.class_id

        # TODO make sure the preprocessing is what we want
        im=PIL.Image.open(im_path)
        im=transforms.ToTensor()(im)
        im=self.crop_head(im,extend=0.1)
        
        if self.dataset=='training':
            im=im.permute(1,2,0).numpy()
            im=self.transform(im)
            im=torch.tensor(im).permute(2,0,1)
        
        im=transforms.Resize(self.resize)(im)
        if self.dataset=='training':
            im = transforms.RandomCrop(self.imsize)(im)
        else:
            im = transforms.CenterCrop(self.imsize)(im)
        im=self.normalization_func(im)

        return im, int(label)
    
    def transform(self,im):
        
        transform = A.Compose([
            A.OneOf([
                A.ToGray(p=0.5),
                A.RandomBrightnessContrast(p=1),
                A.Emboss(p=0.5),
                A.GaussNoise(var_limit=(0,0.1),p=0.5),
                A.MultiplicativeNoise(multiplier=(0.7,1.3), per_channel=True, elementwise=True, always_apply=True, p=1)
                ], p=1),
            A.OneOf([
                A.CoarseDropout(),
                A.Cutout(),
                A.GridDropout(p=0.1),
                A.GridDistortion(p=1),
                A.HorizontalFlip(p=0.5),
            ],p=1),
            A.OneOf([
                A.RandomFog(p=0.5),
                A.GaussianBlur(p=0.5),
                A.Blur(blur_limit=3,p=0.5),
                A.GlassBlur(p=0.5),
                A.OpticalDistortion(),
                A.Sharpen()
            ],p=1),
        ], p=0.95)
        
        im=transform(image=im)
        
        return im['image']
    
    def crop_head(self,im,extend=None):
        corners=torch.stack([im[:,0,0],im[:,0,-1],im[:,-1,0],im[:,-1,-1]])
        background=torch.mode(corners,axis=0).values
        background=background.to(im.device)
        background=torch.all(im.permute(1,2,0)==background,axis=2)

        w=torch.where(torch.all(background,axis=0)==False)[0]
        h=torch.where(torch.all(background,axis=1)==False)[0]
        x1,x2=w.min().item(),w.max().item()
        y1,y2=h.min().item(),h.max().item()

        if type(extend) is float:
            x_max,y_max=im.shape[1],im.shape[2]
            x_center=torch.true_divide(x1+x2,2)
            y_center=torch.true_divide(y1+y2,2)
            x_extend=torch.true_divide(x2-x1,2)*(1+extend)
            y_extend=torch.true_divide(y2-y1,2)*(1+extend)
            x1,x2=max(0,int(x_center-x_extend)),min(int(x_center+x_extend),x_max)
            y1,y2=max(0,int(y_center-y_extend)),min(int(y_center+y_extend),y_max)

        crop=im[:,y1:y2+1,x1:x2+1]

        return crop

#class BFMIdentityDataModule(pl.LightningDataModule):
def BFMIdentity(train_dir, batch_size=256, val_size=0.1, imsize=224, subsample_dataset_fraction=None):

    assert os.path.exists(train_dir), f'path: {train_dir} not found.'

    N_per_individual=363
    class_ids=os.listdir(train_dir)
    all_images=[os.path.join(train_dir, class_id, str(i_image).zfill(3)+'.jpg') for class_id in class_ids for i_image in range(0,N_per_individual)]
    all_ids=[image.split('/')[-2] for image in all_images]

    stage = 'fit'
    if stage == 'fit' or stage is None:

        X_train, X_val, y_train, y_val = train_test_split(all_images, all_ids,
                                                                              stratify=all_ids,
                                                                              test_size=val_size, random_state=42)

        if subsample_dataset_fraction is not None:
            X_train = X_train[:int(len(X_train)*subsample_dataset_fraction)]
            y_train = y_train[:int(len(y_train)*subsample_dataset_fraction)]
            X_val = X_val[:int(len(X_val)*subsample_dataset_fraction)]
            y_val = y_val[:int(len(y_val)*subsample_dataset_fraction)]

        assert len(X_train)==len(y_train)
        assert len(X_val)==len(y_val)

    if stage == 'testing':
        raise NotImplementedError

# def test_dataloader(self):
#     raise NotImplementedError
    
    train_data = BFMIdentityDataset(data_dir=train_dir, im_list=X_train, label_list=y_train, imsize=imsize, dataset='training')
    val_data = BFMIdentityDataset(data_dir=train_dir, im_list=X_val, label_list=y_val, imsize=imsize, dataset='validation')

    return train_data, val_data


def fetch_bfm_dataloaders(args, split=None):
    
    if args.dataset == 'bfm_ids':
        train_dir = '/scratch/nklab/projects/face_proj/datasets/BFM_identity/train'

        train_data, val_data = BFMIdentity(train_dir=train_dir, batch_size=args.batch_size, val_size=0.1, imsize=args.image_size, subsample_dataset_fraction=None)

    elif args.dataset == 'bfm_inv':

        train_dir = '/scratch/nklab/projects/face_proj/datasets/BFM/train_v3'
        train_label_path = '/scratch/nklab/projects/face_proj/datasets/BFM/meta_v3/latents.npy'
        meta_dir = '/scratch/nklab/projects/face_proj/datasets/BFM/meta_v3'

        train_data, val_data = BFMInverse(train_dir, train_label_path, meta_dir, batch_size=args.batch_size, val_size=0.1, imsize=args.image_size, subsample_dataset_fraction=None)


    # def train_dataloader(self, distributed=True):
    if args.distributed:
        sampler=DistributedSampler(train_data, shuffle=False)
    else:
        sampler=None

    print('initializing training dataloader...')
    train_loader =  DataLoader(train_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    # def val_dataloader(self, distributed=True):
    if args.distributed:
        sampler=DistributedSampler(val_data, shuffle=False)
    else:
        sampler=None

    print('initializing validation dataloader...')
    val_loader = DataLoader(val_data, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    return train_loader, val_loader

