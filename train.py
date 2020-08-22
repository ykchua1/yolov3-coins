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
from functools import reduce

CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("start_or_continue", help="indicate to train from start or to continue")
parser.add_argument("epochs", help="how many epochs to train", type=int)
args = parser.parse_args()
assert args.start_or_continue

# set up the neural network
print("Loading network...")
cfgfile = os.path.abspath("cfg/yolov3_mod.cfg") # "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/cfg/yolov3_mod.cfg"
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
    prev_epoch = 0
elif args.start_or_continue == "continue":
    # load state_dict
    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])
    prev_epoch = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]
    
    optimizer = optim.Adam(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    


    
    
    
    

# training loop
if CUDA:
    model.to(torch.device("cuda"))
    
lambda_coord = 0.01
lambda_noobj = 0.1
batch_size = 3
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
imlist, val_im_list = (imlist[:num_train], imlist[num_train:num_train+3])

for epoch in range(epochs):
    print("Starting epoch: {}".format(prev_epoch + epoch))
    
    shuffle(imlist)
    if (len(imlist) % batch_size): leftover = 1
    else: leftover = 0
    loaded_ims = [cv2.imread(x) for x in imlist]
    num_batches = len(imlist) // batch_size + leftover
    im_batches = list(map(prep_image, loaded_ims, [416 for x in range(len(loaded_ims))]))
    im_batches = [torch.cat((im_batches[i*batch_size : min((i+1)*batch_size, len(im_batches))])) for i in range(num_batches)]
    
    val_loaded_ims = [cv2.imread(x) for x in val_im_list]
    val_im_batches = list(map(prep_image, val_loaded_ims, [416 for x in range(len(val_loaded_ims))]))
    val_im_batches = [torch.cat(val_im_batches)]
    if CUDA:
        im_batches = [im_batch.to(torch.device("cuda")) for im_batch in im_batches]
        val_im_batches = [val_im_batches[0].to(torch.device("cuda"))]
    
    for i, batch in enumerate(im_batches):
        loss_sum = [torch.tensor(0.0) for i in range(3)]
        
        fp_list = [imlist[i*batch_size + x][:-4]+".txt" for x in range(batch.shape[0])]
        
        optimizer.zero_grad()
        model.train()
        
        outp = model(batch, CUDA, training=True)
         
        mask1 = create_training_mask_1(outp, fp_list, iou_thresh=0.5)
        mask2 = create_training_mask_2(outp, fp_list, iou_thresh=0.5)

        targ = create_groundtruth(outp, fp_list)
        if CUDA:
            mask1 = mask1.to(torch.device("cuda"))
            mask2 = mask2.to(torch.device("cuda"))
            targ = targ.to(torch.device("cuda"))
        
        sq_err_loss = lambda_coord * mse_loss(torch.clamp(mask1*outp, min=1e-6)[:,:,:4], 416*targ[:,:,:4]) # multiplied by 416 to scale up w/ model output
        cross_entr_loss = cross_entropy(mask1*outp, targ)
        zero_tensor = torch.zeros(outp.shape)
        if CUDA:
            zero_tensor = zero_tensor.to(torch.device("cuda"))
        cross_entr_loss_noobj = lambda_noobj * cross_entropy(torch.clamp(mask2*outp, min=1e-6), zero_tensor)
        
        loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
        loss.backward()
        optimizer.step()
        
        loss_sum = [loss_sum[a] + b for a, b in enumerate([sq_err_loss, cross_entr_loss, cross_entr_loss_noobj])]
        
    loss_mean = [x / num_train for x in loss_sum]
    total_loss_mean = [reduce((lambda x, y: x + y), loss_mean)]
    print("loss means: ", loss_mean)
    print("total loss mean: ", total_loss_mean)
    
    # save to pkl every 5 epoch
    if (prev_epoch + epoch) % 5 == 0:
        torch.save({
            "epoch": prev_epoch + epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss}, "checkpoint.pkl")
    # get the validation losses (code copied training loss section)
    fp_list = [val_im_list[x][:-4]+".txt" for x in range(len(val_im_list))]
    #model.eval()
    outp = model(val_im_batches[0], CUDA, training=True)
    mask1 = create_training_mask_1(outp, fp_list, iou_thresh=0.5)
    mask2 = create_training_mask_2(outp, fp_list, iou_thresh=0.5)
    targ = create_groundtruth(outp, fp_list)
    if CUDA:
        mask1 = mask1.to(torch.device("cuda"))
        mask2 = mask2.to(torch.device("cuda"))
        targ = targ.to(torch.device("cuda"))
    sq_err_loss = lambda_coord * mse_loss(torch.clamp(mask1*outp, min=1e-6)[:,:,:4], 416*targ[:,:,:4]) # multiplied by 416 to scale up w/ model output
    cross_entr_loss = cross_entropy(mask1*outp, targ)
    zero_tensor = torch.zeros(outp.shape)
    if CUDA:
        zero_tensor = zero_tensor.to(torch.device("cuda"))
    cross_entr_loss_noobj = lambda_noobj * cross_entropy(torch.clamp(mask2*outp, min=1e-6), zero_tensor)
    val_loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
    print(val_loss)############################################################################333
    # write to loss log
    with open("loss.txt", "a") as f:
        line = ", ".join([str(float(a)) for a in loss_mean + total_loss_mean])
        line = str(prev_epoch + epoch) + ", " + line
        line = line + ", " + str(float(val_loss))
        f.write(line + "\n")