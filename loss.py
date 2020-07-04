# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 21:32:12 2020

@author: vivek
"""
import torch
import utils

def bce_loss(output, target):
    crit = torch.nn.BCEWithLogitsLoss()
    target = utils.expand_target(target)
    return crit(output,target)*target.size(0)