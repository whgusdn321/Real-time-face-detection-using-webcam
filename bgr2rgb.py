import cv2
from time import * 
import numpy as np
import os
import matplotlib
from math import * 
path = 'C:\\Users\Mythra\Desktop\darkflow-master\\blue_images'







if __name__ == '__main__':
    folder = 'blue_images'
    #print(os.scandir('images')[0])
    img = [im for im in os.listdir(path) if '.png' in im]
    print('img is :',img)
    for image in img:
    	a= cv2.imread(image)
    	cv2.imshow('asdf',a)
    	img_convert= cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    	cv2.imwrite(os.path.join(path, time.strftime("%a%d%H_%M_%S")) +'.png',img_convert) 
   	#bjects = ['head']
    #tl = [(10,10)]
    #br = [(100, 100)]
    #savedir = 'annotations'
    #write_xml(folder, img, objects, tl, br, savedir)
