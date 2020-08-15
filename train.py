import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from util import *
import os
from darknet import Darknet
from random import shuffle

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
lambda_coord = 0.01
lambda_noobj = 0.1
batch_size = 2
epochs = 1
num_train = 21
lr = 0.001

import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss(reduction='sum')

imlist = os.listdir("./data/scattered_coins/")
imlist = list(filter(lambda x: x.split('.')[-1] == "jpg", imlist))
imlist = [os.path.join("./data/scattered_coins/", x) for x in imlist]
imlist = imlist[:num_train]

for epoch in range(epochs):
    print("Starting epoch: {}".format(epoch))
    
    shuffle(imlist)
    if (len(imlist) % batch_size): leftover = 1
    else: leftover = 0
    loaded_ims = [cv2.imread(x) for x in imlist]
    num_batches = len(imlist) // batch_size + leftover
    im_batches = list(map(prep_image, loaded_ims, [416 for x in range(len(loaded_ims))]))
    im_batches = [torch.cat((im_batches[i*batch_size : min((i+1)*batch_size, len(im_batches))])) for i in range(num_batches)]
    
    for i, batch in enumerate(im_batches):
        fp_list = [imlist[i*batch_size + x][:-4]+".txt" for x in range(batch.shape[0])]
        
        optimizer.zero_grad()
        model.train()
        
        inp = model(batch, CUDA, training=True)
        print("Does inp require grad?", inp.requires_grad)
         
        mask1 = create_training_mask_1(inp, fp_list, iou_thresh=0.5)
        mask2 = create_training_mask_2(inp, fp_list, iou_thresh=0.5)

        targ = create_groundtruth(inp, fp_list)
        
        sq_err_loss = lambda_coord * mse_loss((mask1*inp)[:,:,:4], 416*targ[:,:,:4]) # multiplied by 416 to scale up w/ model output
        cross_entr_loss = cross_entropy(mask1*inp, targ)
        cross_entr_loss_noobj = lambda_noobj * cross_entropy(mask2*inp, torch.zeros(inp.shape))
        
        loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
        loss.backward()
        optimizer.step()
        
        print(sq_err_loss, cross_entr_loss, cross_entr_loss_noobj)