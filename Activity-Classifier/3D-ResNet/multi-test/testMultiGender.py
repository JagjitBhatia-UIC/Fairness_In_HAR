# import necessary packages
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import csv
import cvlib as cv
import genderDetectorMulti as gdm

# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = ap.parse_args()

# download pre-trained model file (one-time download)
dwnld_link = "https://github.com/arunponnusamy/cvlib/releases/download/v0.2.0/gender_detection.model"
model_path = get_file("gender_detection.model", dwnld_link,
                     cache_subdir="pre-trained", cache_dir=os.getcwd())

# read input image
image = cv2.imread(args.image)

if image is None:
    print("Could not read input image")
    exit()

# load pre-trained model
model = load_model(model_path)

predictions = gdm.predictGender(model, image)

print('PREDICTIONS: ', predictions)
