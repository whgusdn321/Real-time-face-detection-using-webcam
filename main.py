from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


from math import sqrt
import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)
import time  
from matplotlib import pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'



import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import img_to_array
import h5py
import imutils
cnn = load_model('Gnet.h5py') # Load the CNN we trained previously

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])

static_back = None


while True:
    stime = time.time()
    if static_back is None :
        print('------Video capture has been started------')
        time.sleep(3)
        ret, frame = capture.read()
    else:
        ret, frame = capture.read()
    #make frame more sharpen by using high pass filter
    edge = cv2.filter2D(frame, -1, kernel)

    k = 1
    frame  = frame + (k * edge)

    ################### motion detect part ####################
    # Initializing motion = 0(no motion) 
    motion = 0
    cv2.imshow('frame',frame)
    print('frame.shape is :',frame.shape)
    # Converting color image to gray_scale image 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) 
  
    # Converting gray scale image to GaussianBlur  
    # so that change can be find easily 
    gray = cv2.GaussianBlur(gray, (11, 11), 0) 
  
    # In first iteration we assign the value  
    # of static_back to our first frame 
    if static_back is None: 
        static_back = gray 
        continue

    # Difference between static background  
    # and current frame(which is GaussianBlur) 
    diff_frame = cv2.absdiff(static_back, gray) 

    # current frame is greater than 30 it will show white color(255) 
    thresh_frame = cv2.threshold(diff_frame, 40, 255, cv2.THRESH_BINARY)[1] 
    a = thresh_frame>=1
    print('a.shape: ',a.shape)
    b= thresh_frame[a]
    print('thresh_frame[a].shape',b.shape)
 
    sum = np.sum(thresh_frame)





    #Finding contour of moving object 
    (_, cnts, _) = cv2.findContours(thresh_frame.copy(),  
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
 
    top_coordinates_of_contours = []#(x, y)
    for contour in cnts:       
        y = 9999
        print('contour is :',contour)
        print('contouur size ',contour.shape)
         
        area = cv2.contourArea(contour)
        print('contour area' , area) 
        if cv2.contourArea(contour) < 300: 
            continue #for noise delete                       
        for contour_points in contour: #contour_points format: [[x,y]]
            if y > contour_points[0][1]: #to find a point that has smallest value of y among contour points
                y = contour_points[0][1]
                x = contour_points[0][0]
        top_coordinates_of_contours.append((x,y,contour.shape[0],area))
    
    
    #ROI based on top_coordinates_of_contours:
    ROI_for_search = []
    for top in top_coordinates_of_contours:
        length = int(sqrt(top[3]))
        coordinate = ([top[0]-35, top[1]], [top[0]+35, top[1]+70] , top[2])
        #coordinate = ([top[0]-int(length/2), top[1]], [top[0]+int(length/2), top[1]+length] , top[2], top[3])
        if coordinate[0][0] <0 : #<35, x0
            coordinate[0][0] = 0
            #continue
        elif coordinate[0][1] <0: #<35 y0
        	coordinate[0][1] = 0
            #continue
        elif coordinate[1][0] >= 1024: #>768-35 
        	coordinate[1][0] = 1023
            #continue
        elif coordinate[1][1] >= 768: #>768-35
        	coordinate[1][1] = 767
            #continue
        print('appending coordinate :',coordinate)
        ROI_for_search.append(coordinate)
    	
    for i, candidate_region in enumerate(ROI_for_search):
        
        cv2.rectangle(thresh_frame, tuple(candidate_region[0]),tuple(candidate_region[1]) ,(120,0,0),3)
        #cv2.imshow('for_test',thresh_frame)

        crop_img = gray[candidate_region[0][1] : candidate_region[1][1],
                        candidate_region[0][0] : candidate_region[1][0]]
                        
    	
        
        crop_img_float = cv2.resize(crop_img, (70,70), interpolation = cv2.INTER_CUBIC)
        print('crop_img_float_crazysize ',crop_img_float.shape)
        crop_img_float = crop_img_float/255.0        
        crop_img_float = crop_img_float.reshape((70,70,1))        
        crop_img_float = np.expand_dims(crop_img_float, axis = 0)        
        (face, not_face) = cnn.predict(crop_img_float)[0]
        label = "Face" if face > not_face else 'Not Face'
        area = candidate_region[2]
        print('Face confidence : {}, Not face confidencd : {}'.format(face, not_face))
        cv2.putText(thresh_frame, label , (candidate_region[0][0], candidate_region[0][1]-3),
           	cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0), 2)

        if label == 'Face':
            cv2.putText(thresh_frame, 'Face, score is:'+str(face), (candidate_region[0][0], candidate_region[0][1]-3),cv2.FONT_HERSHEY_COMPLEX, 0.45, (255, 0, 0), 2)


    cv2.imshow('rectengled and thresh_frame',thresh_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()
