# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:29:03 2020

@author: vivek
"""

import cv2
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import functional as F
import yaml
import utils
import torch
from pathlib import Path

with open("config.yaml") as f:
    config = yaml.load(f)

images_dir = Path(config["path"]["images"])
masks_dir = Path(config["path"]["masks"])
data_root = Path(config["path"]["root"])

metadata_csv = data_root.joinpath("comlete_table_with_mcr.csv")
# images_path = images_dir.resolve()
# masks_path = masks_dir.resolve()
metadata_path = metadata_csv.resolve()
images_cat_2 = Path(config["path"]["images_cat"])

#if os.path.exists(images_dir.resolve()):
# img_path_cat = [os.listdir(str(images_dir.resolve()))]
# if images_dir.resolve().joinpath(images_cat_2).exists():
#     img_path_cat.append(os.listdir(str(images_dir.resolve().joinpath(images_cat_2))))

def get_y_path(path):
    """ Get mask path for given image path

    Args:
        path {Path} -- image path
    Returns:
        {Path} -- mask path """
    msk_path = masks_dir.resolve().joinpath(path.parent.stem)
    return msk_path.with_suffix(".png")

def get_filtered_images(df,cond):
    """ Get Paths of filtered images and masks by applying condition to metadata.

    Args:
        df {pd.DataFrame} -- metadata
        condition {bool} -- condition to apply on df
    Returns:
        {list of tuples} -- image and mask paths"""
    img_path = list()
    for i in df[cond].index:
        if str(df["CamId"].iloc[i]) in os.listdir(str(images_dir.resolve())): #img_path_cat[0]:
            img_path.append(images_dir.resolve().
                            joinpath(str(df["CamId"].iloc[i])).
                            joinpath(df["Filename"].iloc[i]))
                                    
        else:
            img_path.append(images_dir.resolve().
                            joinpath(images_cat_2).
                            joinpath(str(df["CamId"].iloc[i])).
                            joinpath(df["Filename"].iloc[i]))
                                    
    msk_path = [get_y_path(img) for img in img_path]
    return list(zip(img_path, msk_path))

class Rescale(object):
    """ Resize Image and Mask.

        if {args is int} preseve aspect ratio  else not
    Args:
        newshape {int, tuple} -- Output Image and Mask height and width
    """
    def __init__(self,newshape):
        assert isinstance(newshape, (int, tuple))
        self.newshape = newshape
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["mask"]

        h,w = image.shape[:2]

        if isinstance(self.newshape, int):
            if h > w:
                new_h, new_w = self.newshape * h / w, self.newshape
            else:
                new_h, new_w = self.newshape, self.newshape * w / h
        else:
            new_h, new_w = self.newshape

        newshape = (int(new_w) , int(new_h))

        image = cv2.resize(image ,  newshape,interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask , newshape , interpolation=cv2.INTER_NEAREST)

        return {"image": image,
                "mask": mask}

class RandomCrop(object):
    """Crop randomly the image and mask

    Args:
        output_size {int, tuple} --  Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size,width_only=False):
        assert isinstance(output_size, (int, tuple))
        self.width_only = width_only
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        left = np.random.randint(0, w - new_w)
        
        if self.width_only:
            #print("img , msk: {}".format(image.shape, mask.shape))
            image = image[:, left: left + new_w, :]
            mask = mask[:, left: left + new_w]
        else:
            top = np.random.randint(0, h - new_h)

            image = image[top: top + new_h,
                            left: left + new_w,:]
            mask = mask[top: top + new_h,
                            left: left + new_w]

        return {'image': image, 'mask': mask}

class Normalize_and_Correct(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std
        #self.inplace = inplace
    def __call__(self,sample):
        sample = utils.correct_shape(sample)
        mask = utils.correct_binary(sample["mask"])
        
        image = torch.from_numpy(sample["image"].astype(np.float32))
        mask = torch.from_numpy(mask)[None,...]

        image = F.normalize(image, self.mean, self.std)[None,...]


        return {"image": image, "mask": mask}

class Compose(object):
    def __init__(self, transforms):
        assert isinstance(transforms , list)
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)    
        return sample