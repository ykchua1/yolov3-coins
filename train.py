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

yolo_type = "tiny"
if yolo_type == "regular":
    yolo_cfg_path = "cfg/yolov3_mod.cfg"
    yolo_weights_path = "yolov3.weights"
elif yolo_type == "tiny":
    yolo_cfg_path = "cfg/yolov3-tiny_mod.cfg"
    yolo_weights_path = "yolov3-tiny.weights"

CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument("start_or_continue", help="indicate to train from start or to continue")
parser.add_argument("epochs", help="how many epochs to train", type=int)
parser.add_argument("--lr", dest='lr', type=float)
args = parser.parse_args()
assert args.start_or_continue

lambda_coord = 0.001
lambda_noobj = 1
batch_size = 21
epochs = args.epochs
lr = 0.001
wd = 1

# set up the neural network
print("Loading network...")
cfgfile = os.path.abspath(yolo_cfg_path) # "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/cfg/yolov3_mod.cfg"
model = Darknet(cfgfile)
if args.start_or_continue == "start":
    weightsfile = os.path.abspath(yolo_weights_path) # "/home/jovyan/work/YOLO_v3_tutorial_from_scratch/yolov3.weights"
    model.load_weights(weightsfile)
print("Network successfully loaded")

# swap out the layers before YOLO and the classes in the YOLO layers
det_layers = get_det_layers(yolo_type=yolo_type)
for i in det_layers:
    in_channels = model.module_list[i-1][0].in_channels
    model.module_list[i-1] = nn.Sequential(nn.Conv2d(in_channels, 27, 1)) 
    model.blocks[i+1]["classes"] = 4
print("Layers have been swapped out")    

if args.start_or_continue == "start":
    # reset the loss log
    with open("loss.txt", "w") as f:
        pass
    prev_epoch = 1
elif args.start_or_continue == "continue":
    # load state_dict
    checkpoint = torch.load("checkpoint.pkl")
    model.load_state_dict(checkpoint["model_state_dict"])
    if CUDA:
        model.to(torch.device("cuda"))
    prev_epoch = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]
    if args.lr:
        print("ON SGD")
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    else:
        print("ON ADAM")
        optimizer = optim.Adam(model.parameters(), weight_decay=wd)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    
    
    
    

# training loop
if args.start_or_continue == "start":
    print("ON ADAM")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.to(torch.device("cuda"))
mse_loss = nn.MSELoss(reduction='sum')
if CUDA:
    mse_loss.to(torch.device("cuda"))

dataset = ImageAnnotationDataset("./data/scattered_coins/train/", transform=transforms.Compose([PrepImage()]))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataset = ImageAnnotationDataset("./data/scattered_coins/val/", transform=transforms.Compose([PrepImage()]), rng=3)
val_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=0)

for epoch in range(epochs):
    print("Starting epoch: {}".format(prev_epoch + epoch))
    
    for i, batch in enumerate(dataloader):
        loss_sum = [torch.tensor(0.0) for i in range(3)]
        if CUDA:
            batch["image"] = batch["image"].to(torch.device("cuda"))
        text_list = batch["text"]
        
        optimizer.zero_grad()
        model.train()
        
        outp = model(batch["image"], CUDA, training=True)
         
        mask1 = create_training_mask_1(outp, text_list, iou_thresh=0.5, yolo_type=yolo_type)
        mask2 = create_training_mask_2(outp, text_list, iou_thresh=0.5, yolo_type=yolo_type)

        targ = create_groundtruth(outp, text_list, yolo_type=yolo_type)
        if CUDA:
            mask1 = mask1.to(torch.device("cuda"))
            mask2 = mask2.to(torch.device("cuda"))
            targ = targ.to(torch.device("cuda"))
        
        sq_err_loss = lambda_coord * mse_loss((mask1*outp)[:,:,:4], 416*targ[:,:,:4]) # multiplied by 416 to scale up w/ model output
        cross_entr_loss = cross_entropy(torch.clamp(mask1*outp, min=1e-6, max=0.9999), targ)
        zero_tensor = torch.zeros(outp.shape).float()
        if CUDA:
            zero_tensor = zero_tensor.to(torch.device("cuda"))
        cross_entr_loss_noobj = lambda_noobj * cross_entropy(torch.clamp(mask2*outp, min=1e-6, max=0.9999), zero_tensor)
        
        loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
        loss.backward()
        optimizer.step()
        
        loss_sum = [loss_sum[a] + b for a, b in enumerate([sq_err_loss, cross_entr_loss, cross_entr_loss_noobj])]
        
    loss_mean = [x / len(dataset) for x in loss_sum]
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
        print("MODEL SAVED")
    # get the validation losses (code copied training loss section)
    for i, batch in enumerate(val_dataloader):
        text_list = batch["text"]
        if CUDA:
            batch["image"] = batch["image"].to(torch.device("cuda"))
        #model.eval()
        outp = model(batch["image"], CUDA, training=True)
        mask1 = create_training_mask_1(outp, text_list, iou_thresh=0.5, yolo_type=yolo_type)
        mask2 = create_training_mask_2(outp, text_list, iou_thresh=0.5, yolo_type=yolo_type)
        targ = create_groundtruth(outp, text_list, yolo_type=yolo_type)
        if CUDA:
            mask1 = mask1.to(torch.device("cuda"))
            mask2 = mask2.to(torch.device("cuda"))
            targ = targ.to(torch.device("cuda"))
        sq_err_loss = lambda_coord * mse_loss((mask1*outp)[:,:,:4], 416*targ[:,:,:4]) # multiplied by 416 to scale up w/ model output
        cross_entr_loss = cross_entropy(torch.clamp(mask1*outp, min=1e-6, max=0.9999), targ)
        zero_tensor = torch.zeros(outp.shape).float()
        if CUDA:
            zero_tensor = zero_tensor.to(torch.device("cuda"))
        cross_entr_loss_noobj = lambda_noobj * cross_entropy(torch.clamp(mask2*outp, min=1e-6, max=0.9999), zero_tensor)
        val_loss = sq_err_loss + cross_entr_loss + cross_entr_loss_noobj
        print(val_loss, cross_entr_loss)############################################################################333
    # write to loss log
    with open("loss.txt", "a") as f:
        line = ", ".join([str(float(a)) for a in loss_mean + total_loss_mean])
        line = str(prev_epoch + epoch) + ", " + line
        line = line + ", " + str(float(val_loss))
        f.write(line + "\n")