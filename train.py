import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from util import *
import os
from darknet import Darknet

# set up the neural network
print("Loading network...")
cfgfile = "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/cfg/yolov3.cfg"
weightsfile = "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/yolov3.weights"
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")

# swap out the layers before YOLO and the classes in the YOLO layers
det_layers = [82, 94, 106]
for i in det_layers:
    in_channels = model.module_list[i-1][0].in_channels
    model.module_list[i-1] = nn.Sequential(nn.Conv2d(in_channels, 27, 1)) 
    model.blocks[i+1]["classes"] = 4
print("Layers have been swapped out")    

# training loop
CUDA = torch.cuda.is_available()

lambda_coord = 1
lambda_noobj = 1

import torch.optim as optim
mse_loss = nn.MSELoss(reduction='sum')

imlist = os.listdir("./data/scattered_coins/")
imlist = list(filter(lambda x: x.split('.')[-1] == "jpg", imlist))
imlist = [os.path.join("./data/scattered_coins/", x) for x in imlist]

for epoch in range(2):
    
    for i in imlist:
        fp_list = [i[:-4]+".txt"]
        
        loaded_imgs = [cv2.imread(i)]
        img_batch = list(map(prep_image, loaded_imgs, [416 for x in range(len(loaded_imgs))]))
        img_batch = img_batch[0]
        inp = model(img_batch, CUDA)
         
        mask1 = create_training_mask_1(inp, fp_list, iou_thresh=0.5)
        mask2 = create_training_mask_2(inp, fp_list, iou_thresh=0.5)

        tar = create_groundtruth(inp, fp_list)
        
        sq_err_loss = lambda_coord * mse_loss((mask1*inp)[:,:,:4], 416*tar[:,:,:4])
        cross_entr_loss = cross_entropy(mask1*inp, tar)
        cross_entr_loss_noobj = lambda_noobj * cross_entropy(mask2*inp, torch.zeros(inp.shape))
        
        print(sq_err_loss, cross_entr_loss, cross_entr_loss_noobj)