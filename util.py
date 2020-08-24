from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):

    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors
    
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False
    


    for ind in range(batch_size):
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS
    
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4], as_tuple=False))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        
        for cls in img_classes:
            #perform NMS

        
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2], as_tuple=False).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4], as_tuple=False).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def transform_b2t(pred, anchors=[(39,38),  (45,44),  (49,49),  (70,36),  (54,53),  (58,58),  (64,62),  (69,68),  (76,75)]): 
    """
    pred should be a tensor of shape (batches * 10647 * (5 + num_classes))
    pred coordinates have ranges of 13*13, 26*26 and 52*52
    
    """
    batch_size = pred.shape[0]
    
    # x and y coordinates transformation
    cx1 = torch.tensor([x % 13 for x in range(13**2)]).float()
    cx2 = torch.tensor([x % 26 for x in range(26**2)]).float()
    cx3 = torch.tensor([x % 52 for x in range(52**2)]).float()
    cx = torch.cat([cx1, cx2, cx3]).repeat_interleave(3)
    
    cy1 = torch.tensor([x // 13 for x in range(13**2)]).float()
    cy2 = torch.tensor([x // 26 for x in range(26**2)]).float()
    cy3 = torch.tensor([x // 52 for x in range(52**2)]).float()
    cy = torch.cat([cy1, cy2, cy3]).repeat_interleave(3)
    
    c = torch.cat([cx.unsqueeze(1), cy.unsqueeze(1)], 1) # outputs 10647 * 2 tensor
    c = c.repeat(batch_size, 1, 1) # outputs batches * 10647 * 2 tensor
    
    pred[:,:,:2] = torch.log((pred[:,:,:2] - c) / (1 - pred[:,:,:2] + c))
    
    # width and height dimension transformation
    stride1 = torch.tensor([32 for i in range(3*13**2)])
    stride2 = torch.tensor([16 for i in range(3*26**2)])
    stride3 = torch.tensor([8 for i in range(3*52**2)])
    strides = torch.cat([stride1, stride2, stride3], 0).float() # size 10647 tensor
    
    pw = torch.tensor([]).float()
    ph = torch.tensor([]).float()
    for i, dim in enumerate([13, 26, 52]):
        pw_tmp = torch.tensor([x[0] for x in anchors[6-3*i : 9-3*i]]).float().repeat(dim**2)
        pw = torch.cat([pw, pw_tmp], 0)
        
        ph_tmp = torch.tensor([x[1] for x in anchors[6-3*i : 9-3*i]]).float().repeat(dim**2)
        ph = torch.cat([ph, ph_tmp], 0)
    pw /= strides
    ph /= strides
    p = torch.cat([pw.unsqueeze(1), ph.unsqueeze(1)], 1) # outputs 10647 * 2 tensor
    p = p.repeat(batch_size, 1, 1) # outputs batches * 10647 * 2 tensor
    
    pred[:,:,2:4] = torch.log(pred[:,:,2:4] / p)
    
    return pred

def create_objbb_dict(fp, iou_thresh=0.5, yolo_type="regular"):
    """
    returns objbb_dict
    
    """
    
    def find_row(x, y, stride=32): # gets the row number of the first bounding box assigned to the xy position
        assert stride in (32, 16, 8)
        baserow = {32: 0, 16: 3*13**2, 8: 3*13**2+3*26**2}[stride]
        
        if stride == 32:
            x_index = x // (1/13)
            y_index = y // (1/13)
            cell_index = 13*y_index + x_index
            rel_row = cell_index * 3
            return baserow + rel_row
        
        if stride == 16:
            x_index = x // (1/26)
            y_index = y // (1/26)
            cell_index = 26*y_index + x_index
            rel_row = cell_index * 3
            return baserow + rel_row
        
        if stride == 8:
            x_index = x // (1/52)
            y_index = y // (1/52)
            cell_index = 52*y_index + x_index
            rel_row = cell_index * 3
            return int(baserow + rel_row)
        
    def find_iou(x, y, w, h, row): # bb is the bounding box number (0, 1 or 2)
        if row < 3*13**2:
            stride = 32
        elif row < 3*13**2 + 3*26**2:
            stride = 16
        else:
            stride = 8
        
        bb = row % 3 
        
        if yolo_type == "regular": strd2anchind = {32: 6, 16: 3, 8: 0}
        elif yolo_type == "tiny": strd2anchind = {32: 3, 16: 0}
        anchor_index = strd2anchind[stride] + bb
        anchor_index = int(anchor_index)
        filter_dim = {32: 13, 16: 26, 8: 52}[stride]
        x_coord416 = (x // (1/filter_dim)) * float(stride)
        y_coord416 = (y // (1/filter_dim)) * float(stride)
        
        anchors = get_anchors(yolo_type=yolo_type)
        
        b1x1, b1x2 = ((x - w/2)*416.0, (x + w/2)*416.0)
        b1y1, b1y2 = ((y - h/2)*416.0, (y + h/2)*416.0)
        b2x1, b2x2 = (x_coord416 - anchors[anchor_index][0]/2, x_coord416 + anchors[anchor_index][0]/2)
        b2y1, b2y2 = (y_coord416 - anchors[anchor_index][1]/2, y_coord416 + anchors[anchor_index][1]/2)
        
        box1 = torch.tensor([[b1x1, b1y1, b1x2, b1y2]]).clamp(min=0, max=416) # box1 is the groundtruth bounding box
        box2 = torch.tensor([[b2x1, b2y1, b2x2, b2y2]]).clamp(min=0, max=416) # box2 is the anchor
        
        iou = bbox_iou(box1, box2)
        
        return iou
        
    with open(fp) as f:
        lines = f.readlines()
        
    objbb_dict = {}
    for i, line in enumerate(lines): # iterate thru objects
        sp = line.split()
        
        x, y, w, h = (float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4]))
        bounding_box_rows = [find_row(x, y, stride=32) + bb for bb in range(3)]
        bounding_box_rows += [find_row(x, y, stride=16) + bb for bb in range(3)]
        if yolo_type == "regular":
            bounding_box_rows += [find_row(x, y, stride=8) + bb for bb in range(3)] # bounding_box_rows is len 9
        objbb_dict[i] = [int(sp[0]), [(int(row), find_iou(x, y, w, h, row)) for row in bounding_box_rows]]
        
        objbb_dict[i][1].sort(key = lambda x: x[1], reverse=True) # sort the bbs by iou (reversed) 
        objbb_dict[i][1][1:] = list(filter(lambda x: x[1] > iou_thresh, objbb_dict[i][1][1:])) # filters the bbs by threshold
        
        objbb_dict[i].append((x, y, w, h))
        
    return objbb_dict

def create_training_mask_1(pred, fp_list, iou_thresh=0.5, yolo_type='regular'): # mask 1 allows detection rows only
    """
    takes in batches * 10647 * (5 + num_classes) tensor as input
    
    returns a training mask tensor
    
    fp is the file path the annotation data
    
    """
    
    assert len(fp_list) == int(pred.shape[0])
    
    training_mask = torch.zeros(pred.shape).float()
    
    for i, fp in enumerate(fp_list):
        objbb_dict = create_objbb_dict(fp, iou_thresh=iou_thresh, yolo_type=yolo_type)
    
        for key, value in objbb_dict.items():
            det_row = value[1][0][0] # row of assigned bounding box
            training_mask[i,det_row,:] = 1
        
    return training_mask

def create_training_mask_2(pred, fp_list, iou_thresh=0.5, yolo_type="regular"): # mask 2 enables no object loss
    """
    takes in batches * 10647 * (5 + num_classes) tensor as input
    
    returns a training mask tensor
    
    fp is the file path the annotation data
    
    """
    
    assert len(fp_list) == int(pred.shape[0])
     
    training_mask = torch.zeros(pred.shape).float()
    training_mask[:,:,4] = 1
    
    for i, fp in enumerate(fp_list):
        objbb_dict = create_objbb_dict(fp, iou_thresh=iou_thresh, yolo_type=yolo_type)
        
        for key, value in objbb_dict.items():
            num_valid_bb = len(value[1]) # the number of valid bounding boxes for the object
            
            for j in range(num_valid_bb):
                sec_row = value[1][j][0] # secondary rows consist of all valid bbs (best bb or iou above thresh)
                training_mask[i,sec_row,4] = 0. # remove no-object loss for boxes above iou threshold
    
    return training_mask

def create_groundtruth(pred, fp_list, yolo_type="regular"):
    """
    returns groundtruth tensor (batches * 10647 * (5 + num_classes))
    
    groundtruth tensor coordinate values have range [0, 1]
    
    """
    assert len(fp_list) == int(pred.shape[0])
    
    groundtruth = torch.zeros(pred.shape).float()
    
    for i, fp in enumerate(fp_list):
        objbb_dict = create_objbb_dict(fp, yolo_type=yolo_type)
        
        for key, value in objbb_dict.items():
            det_row = value[1][0][0]

            groundtruth[i,det_row,:4] = torch.tensor(value[2]) # value[2] is tuple (x, y, w, h)
            groundtruth[i,det_row,4] = 1 # assign is_object
            groundtruth[i,det_row,int(5+value[0])] = 1 # assign class
        
    return groundtruth

def cross_entropy(outp, tar):
    output = -(tar[:,:,4:]*torch.log(outp[:,:,4:]) + (1 - tar[:,:,4:])*torch.log(1 - outp[:,:,4:]))
    output = torch.sum(output)
    return output

def get_anchors(yolo_type="regular"):
    if yolo_type == "regular":
        return [(39,38),  (45,44),  (49,49),  (70,36),  (54,53),  (58,58),  (64,62),  (69,68),  (76,75)]
    elif yolo_type == "tiny":
        return [(41,39), (47,46), (53,52), (58,57), (65,63), (74,73)]
    
def get_det_layers(yolo_type="regular"):
    if yolo_type == "regular":
        return [82, 94, 106]
    elif yolo_type == "tiny":
        return [16, 23]