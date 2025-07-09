import os
import cv2
import detection
import numpy as np
import paddleocr
import image_processing
from paddleocr import PaddleOCR, draw_ocr


def ref_num_extraction(image_path, model_path):

    directory = 'reference_numbers'
    if not os.path.exists(directory):
        os.makedirs(directory)

    save_path = './reference_numbers/'

    ocr = PaddleOCR(use_angle_cls=False, lang='en')
    reference_numbers_list = []
    
    segmented_land_list, coordinates_list = image_processing.segmented_land_extraction(image_path, model_path)

    for i in range(len(segmented_land_list)):
        image = cv2.cvtColor(segmented_land_list[i], cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(segmented_land_list[i],cls=True)

        full_reference_num = ''
        if len(result[0])==0:
            full_reference_num = 'Ref. No. unrecognized'

        for item in result[0]:
            full_reference_num += str(item[-1][0]) + ' '
        reference_numbers_list.append(full_reference_num)

        file_name = 'reference_numbers'+str(i)+'.txt'
        file_path = os.path.join(save_path, file_name)         

        with open(file_path, 'w') as file:
            file.write(reference_numbers_list[i])
            
    return coordinates_list, reference_numbers_list
