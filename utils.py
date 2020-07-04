# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:30:15 2020

@author: vivek
"""
import numpy as np
import matplotlib.image as mpimg
import torch
import matplotlib.pyplot as plt
import cv2

def expand_target(image, opp = False):
    """
        Expand the shape of Target : -1 , C , H , W --> -1 , C* , H , W

    :param image: Batch {Tensor}
    :param opp: If True , Do the reverse of expand
    :return: Expanded Shape {Tensor}
    """

    if opp:
        return image[:,0:1,:,:]
    else:
        image_0 = image
        image_1 = torch.where(image == 1 , torch.zeros_like(image) , torch.ones_like(image))
        image = torch.cat([image_0, image_1] , dim = 1)
    return image

def correct_shape(sample):
    """" Image from CHW (HWC) to HWC (CHW).
        and Mask from HW to CHW """
    image = np.transpose(sample["image"] , (2,0,1))
    mask = sample["mask"][None,...]
    return {"image":image,
            "mask":mask}

def denormalize(tensor,mean,std):
    """ Denormalize torch tensor"""
    for t , m, s in zip(tensor,mean,std):
        t.mul_(s).add_(m)
    return tensor

def get_horizon(mask):
    """Get Horizon Coordinates in Mask"""
    if len(mask.shape) == 3:
        mask = mask.squeeze()
    horizon = list()

    for col in range(0,mask.shape[1]):
        for row in range(1,mask.shape[0]):
            if mask[row,col] != mask[row-1,col]:
                horizon.append([row,col])
    return horizon

def show_images_from_path(img_path,msk_path,fig_size=(10,4)):
    """show image and mask given paths"""
    try:
        img = mpimg.imread(img_path)
    except:
        print("File not found!")
        return None
    msk = mpimg.imread(msk_path,0)
    fig, ax = plt.subplots(1,2, figsize=fig_size)
    ax[0].set_title("Image")
    ax[1].set_title("Mask")
    ax[0].imshow(img)
    ax[1].imshow(msk)
    
def correct_binary(image , opp = False):
    if opp:
        image = np.where(image == 1, 255, 0)
    else:
        image = np.where(image == 255 , 1 ,0)
    return image

def show_sample_from_dl(dl,denorm=False,figsize=(10,4)):
    sample = next(iter(dl))
    idx = np.random.randint(0,sample["images"].shape[0])
    img = np.copy(sample["images"].numpy())
    
    img = torch.from_numpy(img)[idx]
    #print(img.max(),img.min())
    ##img = denormalize(img, mean, std)
    #print(img.max(),img.min())

    msk = sample["masks"].cpu().numpy()[idx]
    
    img = np.transpose(img , (1,2,0))
    img = np.int64(cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2RGB))
    msk = np.squeeze(msk,0)
    #print(msk.shape)
    #msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
    msk = correct_binary(msk , True)
    fig, ax = plt.subplots(1,2,figsize=figsize)
    ax[0].imshow(img)
    ax[1].imshow(msk)