import cv2
from time import * 
import numpy as np
import os
import matplotlib
from math import * 
#This code is to generate face image datasets by capturing images...
#if person is passing by over the window, the cam will capture those images
#and save them into directory(path) 

#key variable = difference : this is a value that determines how sensitively 
#detect the movement(changes in the array)

#input : None(There is no input)
#output : None(There is no output)
path = 'C:\\Users\Mythra\Desktop\darkflow-master\capture_images'

video = cv2.VideoCapture(0)
a = -1
pre_frame = np.ndarray(shape = (480,640, 3),  dtype = int)
#pre_difference = 0
while True:
	sleep(0.1)
	a = a+1

	#create a frame object
	pre_time = time()
	
	check, frame = video.read()		
	
	post_frame = np.array(frame)
	if a==0 :
		pre_frame = np.array(frame)
	difference = np.sum(np.power(post_frame - pre_frame ,2))/1000
	
	if a==0 or a== 1:
		pre_difference = difference
	difference_gradient = difference - pre_difference
	
	#print('difference is :' , difference)
	print('difference_gradient is = ',abs(difference_gradient))
	
	#if difference > 20000000 -> moving -> should be captured and download!
	cv2.imshow("Captureing", frame)
	
	if difference_gradient >4000:
		#frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		filename = os.path.join(path, strftime("%a%d%H_%M_%S")) +'.png' 
		print(filename)
		cv2.imwrite(filename, frame)
		
	
	pre_frame = post_frame
	pre_difference = difference 
	
	post_time = time()
	print('FPS : {:.2f}'.format(1/(post_time - pre_time)))

	key = cv2.waitKey(1)

	if key == ord('q'):
		break

print(a)


#shutdown the camera
video.release()