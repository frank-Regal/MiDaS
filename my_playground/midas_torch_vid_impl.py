#!/usr/bin/bash python3

import timm
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

VIDEO_PATH='/project/ws_dev/src/hri_cacti_xr/data/005_data/devices/mono_rf/raw/MoveForward/S0507164414-D003-C008.avi'

"""
MiDAS Setup
"""
#model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)

if (torch.cuda.is_available()):
    print("Cuda Available")
    device = torch.device("cuda")
else:
    print("Cuda not available")
    device = torch.device("cpu")

midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

"""
Prediction
"""
# open video
vid = cv2.VideoCapture(VIDEO_PATH)

# create output video
w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(vid.get(cv2.CAP_PROP_FPS))
print(f'[Input Video Properties] Width: {w} pixels; Height: {h} pixels; FPS: {fps}')

depth_pred_filename = '/project/ws_dev/src/hri_cacti_xr/gesture_recognition/research/MiDaS/my_playground/thres_pred_norm-' + model_type + '.mp4'
depth_vid = cv2.VideoWriter(depth_pred_filename,  
                         cv2.VideoWriter_fourcc('m','p','4','v'), 
                         fps, (w,h)) 
# set global vars
#min_depth_ = float('inf')
#max_depth_ = float('-inf')

min_depth_ = -267.038
max_depth_ = 6612.6973

# loop through each frame in the video
while (vid.isOpened()):
    
    # read a frame
    ret, frame = vid.read()
    
    if ret:
        # convert to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # send to GPU
        input_batch = transform(img).to(device)

        # make predictions
        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # normalize and threshold the predictions
        depth_map = prediction.cpu().numpy()
        
        # Update global min and max depth values
        #min_depth_ = min(min_depth_, np.min(depth_map))
        #max_depth_ = max(max_depth_, np.max(depth_map))
        
        # normalize the video 
        normalized_depth = (depth_map - min_depth_) / (max_depth_ - min_depth_)
        depth_pred = (normalized_depth * 255).astype(np.uint8)
        
        #depth_pred = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        #depth_pred = cv2.convertScaleAbs(depth_pred)
        thres_mask = cv2.inRange(depth_pred, 20, 255)

        # write to video
        depth_image_cv = cv2.cvtColor(thres_mask, cv2.COLOR_GRAY2RGB)
        depth_vid.write(depth_image_cv)
    
    else:
        break

# close video
vid.release()
depth_vid.release()


print(f"min depth: ", min_depth_)
print(f"max depth: ", max_depth_)



# Normalize depth map using global min and max depth values
#normalized_depth = (depth_map - global_min_depth) / (global_max_depth - global_min_depth)