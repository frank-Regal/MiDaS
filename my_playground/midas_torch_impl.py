#!/usr/bin/bash python3

import timm
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

image = '/project/ws_dev/src/hri_cacti_xr/data/004_data/imgs/deictic_right_hand_v1/img_00022.jpg'

model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
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

img = cv2.imread(image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

depth_pred = prediction.cpu().numpy()
depth_pred = cv2.normalize(depth_pred, None, 0, 255, cv2.NORM_MINMAX)
depth_pred = cv2.convertScaleAbs(depth_pred)
# depth_pred = cv2.medianBlur(depth_pred, 11)
thres_mask = cv2.inRange(depth_pred, 180, 255)
#result = cv2.bitwise_and(depth_pred, depth_pred, mask=thres_mask)

depth_pred_filename = '/project/ws_dev/src/hri_cacti_xr/gesture_recognition/research/MiDaS/my_playground/depth_pred-' + model_type + '.jpeg'
thres_mask_filename = '/project/ws_dev/src/hri_cacti_xr/gesture_recognition/research/MiDaS/my_playground/thres_mask-' + model_type + '.jpeg'
cv2.imwrite(depth_pred_filename,depth_pred)
cv2.imwrite(thres_mask_filename,thres_mask)