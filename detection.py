import cv2
import numpy as np
from ultralytics import YOLO
from ultralyticsplus import render_result


def land_detection(image_path, model_path):

    image_path = image_path
    img = cv2.imread(image_path)
    model_path = model_path
    model = YOLO(model_path)  # load a pretrained model (recommended for training)
    results = model(image_path, save=True)
    
    return img, results