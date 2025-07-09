import os
import cv2
import detection
import numpy as np
from ultralytics import YOLO
from ultralyticsplus import render_result


def segmented_land_extraction(image_path, model_path):
    
    mask_list = []
    coordinates_list = []
    segmented_land_list = []
    
    directory = 'segmented_land_images'
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    img, results = detection.land_detection(image_path, model_path)
    
    i=0
    for result in results:
        for mask, box in zip(result.masks.data , result.boxes.xyxy):
            mask = mask.numpy().astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            coordinates_list.append(np.argwhere(mask).tolist())
            mask_list.append(mask)

            boxes = box.tolist()
            (x1,y1,x2,y2) = (int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3]))
            masked = cv2.bitwise_and(img, img, mask=mask)
            segmented_land = masked[y1:y2, x1:x2]
            segmented_land_list.append(segmented_land)

            cv2.imwrite('./segmented_land_images/segmented_land'+str(i)+'.png', segmented_land)
            i+=1
    
    return segmented_land_list, coordinates_list