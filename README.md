# HMLR_Computer_Vision
HM Land Registry mini-challenge on Land-Marking Detection and Segmentation using Deep Learning

The HMLR_Computer_Vision.ipynb notebook can be used to view the entire steps and their results for the project.

Python 3.7.5 is required for this project
First create a virtual environment and activate it.
Next, install the required packages to virtual environment using the following command:
python -m pip install -r requirements.txt

After installation of packages run the project using the following command on the terminal:
python image_to_geopackage.py

There are four python scripts for this project:

detection.py: which uses the trained model on detection and segmentation

image_processing.py: which uses the results from detection.py to perform image_processing on the results and extract the segmented/detected images of land

ocr.py: which uses the results from image_processing.py to extract the reference numbers written on the land segments
image_to_geopackage.py: which converts the segmented land images to geopackages and uses the text from ocr.py to inlcude them as meta-information to the geopackages. 

The deep learning model weights which perform object detection and segmentation are present in last.pt in the following directory:
runs/segment/train15/weights/

The results for model training are located in the following directory:
runs/segment/train15/weights/

The result from detection.py is located in the following directory:
runs/segment/predict/

The results from image_processing.py are located in the following directory:
segmented_land_images

The results from ocr.py are located in the following directory:
reference_numbers

The results from image_to_geopackage.py are located in the following directory:
geopackage_files






