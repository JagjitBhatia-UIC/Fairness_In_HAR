# author: Arun Ponnusamy
# website: https://www.arunponnusamy.com

# import necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils import get_file
import numpy as np
import argparse
import cv2
import os
import csv
import cvlib as cv

def getPredAccuracy(preds, actuals, confidences):
    # Ensure equal number of predictions and actuals
    if(len(preds) != len(actuals)):
        print("ERROR: Must be equal number of predictions and results")
        return 0
    
    count = 0

    for i in range(0, len(preds)):
        predval = 0
        if confidences[i][1] > confidences[i][0]: predval = 1
        print("-- TEST #", i, " PREDICTION: ", preds[i], " CONFIDENCE: ", str(round(confidences[i][predval], 2)), "%   , EXPECTED: ", actuals[i][0])
        
        if preds[i] == actuals[i][0]: count += 1
            
    
    return ((count/len(preds)) * 100)

def getAverageConf(confs):
    sum = 0
    for c in confs:
        sum += max(c)
    return sum/len(confs)

# Assuming only one face - if more than one, only first face detected will be used
def predictGender(model, image):
    # detect faces in the image
    face, confidence = cv.detect_face(image)
    classes = ['man','woman', 'unknown']

    if len(face) < 1:
        return classes[2]
    
    predictions = []

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle       
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # draw rectangle over face
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(image[startY:endY,startX:endX])

        # preprocessing for gender detection model
        try:
            face_crop = cv2.resize(face_crop, (96,96))
        except cv2.error as e:
            print('Invalid frame!')
            predictions.append(classes[2])
            continue
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # apply gender detection on face
        conf = model.predict(face_crop)[0]

        norm = [(float(i)/sum(conf)) * 100 for i in conf]

        if max(norm) < 65:
            predictions.append(classes[2])

        elif conf[1] > conf[0]:
            predictions.append(classes[1])
        
        else:
            predictions.append(classes[0])
    
    return predictions






   
