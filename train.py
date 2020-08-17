import time
import torch
import torch.nn as nn
import numpy as np
import cv2
from util import *
import os
from darknet import Darknet
from random import shuffle
import argparse
import torch.optim as optim

CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("start_or_continue", help="indicate to train from start or to continue")
parser.add_argument("epochs", help="how many epochs to train", type=int)
args = parser.parse_args()
assert args.start_or_continue

# set up the neural network
print("Loading network...")
cfgfile = os.path.abspath("cfg/yolov3.cfg") # "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/cfg/yolov3.cfg"
model = Darknet(cfgfile)
if args.start_or_continue == "start":
    weightsfile = os.path.abspath("yolov3.weights") # "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/yolov3.weights"
    model.load_weights(weightsfile)
print("Network successfully loaded")

# swap out the layers before YOLO and the classes in the YOLO layers
det_layers = [82, 94, 106]
for i in det_layers:
    in_channels = model.module_list[i-1][0].in_channels
    model.module_list[i-1] = nn.Sequential(nn.Conv2d(in_channels, 27, 1)) 
    model.blocks[i+1]["classes"] = 4
print("Layers have been swapped out")    

if args.start_or_continue == "start":
    # reset the loss log
    with open("loss.txt", "w") as f:
        pass
elif args.start_or_continue == "continue":
    # load state_dict
    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    


    
    
    
    

# training loop
if CUDA:
    model.to(torch.device("cuda"))
    
lambda_coord = 0.01
lambda_noobj = 0.1
batch_size = 2
epochs = args.epochs
num_train = 21
lr = 0.001

optimizer = optim.Adam(model.parameters(), lr=lr)
mse_loss = nn.MSELoss(reduction='sum')
if CUDA:
    mse_loss.to(torch.device("cuda"))

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
    if CUDA:
        im_batches = [im_batch.to(torch.device("cuda")) for im_batch in im_batches]
    
    for i, batch in enumerate(im_batches):
        fp_list = [imlist[i*batch_size + x][:-4]+".txt" for x in range(batch.shape[0])]
        
        optimizer.zero_grad()
        model.train()
        
        inp = model(batch, CUDA, training=True)
         
        mask1 = create_training_mask_1(inp, fp_list, iou_thresh=0.5)
        mask2 = create_training_mask_2(inp, fp_list, iou_thresh=0.5)

        targ = create_groundtruth(inp, fp_list)
        if CUDA:
            mask1 = mask1.to(torch.device("cuda"))
            mask2 = mask2.to(torch.device("cuda"))
            targ = targ.to(torch.device("cuda"))
        
        sq_err_loss = lambda_coord * mse_loss((mask1*inp)[:,:,:4], 416*targ[:,:,:4]) # multiplied by 416 to scale up w/ model output
        cross_entr_loss = cross_entropy(mask1*inp, targ)
        zero_tensor = torch.zeros(inp.shape)
        if CUDA:
            zero_tensor = zero_tensor.to(torch.device("cuda"))
        cross_entr_loss_noobj = lambda_noobj * cross_entropy(mask2*inp, zero_tensor)
        
        loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
        loss.backward()
        optimizer.step()
        
    print(sq_err_loss, cross_entr_loss, cross_entr_loss_noobj)
    
    # save to pkl every epoch
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss}, "checkpoint.pkl")
    # write to loss log
    with open("loss.txt", "a") as f:
        total_loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
        line = ", ".join([str(int(a)) for a in (sq_err_loss, cross_entr_loss, cross_entr_loss_noobj, total_loss)])
        f.write()