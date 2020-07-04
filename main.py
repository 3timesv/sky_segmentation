# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:34:17 2020

@author: vivek
"""

import download_data as dd
import preprocess as pp
import sky_dataset as sd
import yaml
from torch.utils.data import DataLoader
import random
import pandas as pd
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import time
import loss
from pathlib import Path
import functional as F
import os


with open("config.yaml") as f:
    config = yaml.load(f)

images_dir = config["path"]["images"]
masks_dir = config["path"]["masks"]
data_root = config["path"]["root"]

# Download Data
print("==== Downloading data...")
dd.download_skyf_data()
print("==== Downloaded")

# Get DataLoaders
print("==== Preparing Dataset...")
metadata = pd.read_csv( os.path.join(data_root,"complete_table_with_mcr.csv"))

condition = metadata["Fog"] == 0

path_list = pp.get_filtered_images(metadata, condition)
shuffled_path_list = random.sample(path_list, len(path_list))

train_split = config["train"]["train_split"]

train_list = shuffled_path_list[: int(len(path_list)*train_split) ]
val_list = shuffled_path_list[int(len(path_list)*train_split) : ]

mean = config["train"]["normalize"]["imagenet"]["mean"]
std = config["train"]["normalize"]["imagenet"]["std"]

transform = {"train" : pp.Compose([pp.Rescale((128,200)),
                     pp.RandomCrop(128,True),
                     pp.Normalize_and_Correct(mean,std)]),
            "val" : pp.Compose([pp.Rescale((128,128)),
                                pp.Normalize_and_Correct(mean,std)]) }

train_ds = sd.dataset(train_list,images_dir,transform["train"])
val_ds = sd.dataset(val_list,images_dir , transform["val"])

batch_size = config["train"]["batch_size"]

train_dl = DataLoader(train_ds,
                     batch_size=batch_size,
                     shuffle=True,
                     collate_fn= sd.collate)

val_dl = DataLoader(val_ds,
                    batch_size= batch_size,
                    collate_fn= sd.collate)
print("==== Prepared")

# Model and Optimizer
print("==== Define Model and Optimizer...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num_class = 2
model = smp.Unet('resnet34', encoder_weights='imagenet',classes=num_class).to(device)

# freeze backbone layers
#for l in model.base_layers:
#    for param in l.parameters():
#        param.requires_grad = False

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=8, gamma=0.1)
print("==== Defined")

# train
print("==== Start training...")
epochs = config["train"]["epochs"]
save_path = config["train"]["save_path"]

best_loss = 1e10

for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch , epochs-1))
    print("=="*10)
    start = time.time()

    model.train()
    sum_loss = 0.0
    num = 0.0
    for sample in train_dl:
        inputs = sample["images"].to(device)
        target = sample["masks"].to(device)
        optimizer_ft.zero_grad()
        output = model(inputs)
        train_loss = loss.bce_loss(output, target.float())
        train_loss.backward()
        optimizer_ft.step()

        sum_loss += train_loss
        num += inputs.size(0)
    print("Train loss ",sum_loss/num)

    model.eval()
    sum_loss = 0.0
    num = 0.0
    for sample in val_dl:
        inputs = sample["images"].to(device)
        target = sample["masks"].to(device)
        optimizer_ft.zero_grad()
        output = model(inputs)
        val_loss = loss.bce_loss(output, target.float())

        sum_loss += val_loss
        num += inputs.size(0)
    print("Validation Loss ", sum_loss/num)

    if sum_loss < best_loss:
        print("Saving Best Model")
        best_loss = sum_loss
        torch.save(model, save_path)
    print("Epoch time ",(time.time() - start)//60,"minutes")
    print("Best Validation Loss ", best_loss)
