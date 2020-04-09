
##data computation for faceSrub

import numpy as np
import urllib
import urllib.request
import cv2

from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import h5py
import matplotlib.pyplot as plb


def url_to_image(url):
  try:
     resp = urllib.request.urlopen(url, timeout=5)
  except urllib.error.HTTPError as e:
     return np.zeros(2,3)

  image = np.asarray(bytearray(resp.read()), dtype="uint8")
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  if(image.all() == None):
    return np.zeros(2,3)

  return image

	
	

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r'C:\Users\duo li\Desktop\third year\aps360\project\datacomputation\shape_predictor_5_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
hf = h5py.File(r'C:\Users\duo li\Desktop\third year\aps360\project\datacomputation\data_actor.h5','r+')
img = hf.create_dataset("train_img", [40000,224,224,3], dtype= np.uint8)
name = hf.create_dataset("names",(40000,), dtype=h5py.string_dtype())

def main():
	i=0
	j=1024
		
	for line in open(r'C:\Users\duo li\Desktop\third year\aps360\project\datacomputation\facescrub_actors.txt'):

		if (i>=0):
			#print(line)

			try:
				image = url_to_image(line.split(maxsplit=7)[4])
			except:
				print(i)
				i+=1
				continue
			
			
			x1=max(0,int(line.split(maxsplit=7)[5].split(',')[0])-40)
			y1=max(0,int(line.split(maxsplit=7)[5].split(',')[1])-40)
			x2=max(0,int(line.split(maxsplit=7)[5].split(',')[2])+40)
			y2=max(0,int(line.split(maxsplit=7)[5].split(',')[3])+40)
			if(np.shape(image)[1]<x1 or np.shape(image)[1]<x2 or np.shape(image)[0]<y1 or np.shape(image)[0]<y2 ):
				print(i)
				i+=1
				continue
		
			scale_percent = 60 # percent of original size
			width = int(img.shape[1] * scale_percent / 100) 
			height = int(img.shape[0] * scale_percent / 100) 
			dim = (width, height) 

			resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
			gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

			# show the original input image and detect faces in the graysca
			rects = detector(gray, 2)

			# loop over the face detections
			for rect in rects:
			  faceAligned = fa.align(image, gray, rect)
			  faceFinal = imutils.resize(faceAligned[60:196,60:196], width=224)
			  name=line.split(maxsplit=7)[0]+' '+line.split(maxsplit=7)[1]+' '+line.split(maxsplit=7)[2]
			  print("save:", name)
			 		 
			  hf["train_img"][j,...]=faceFinal
			  hf["names"][j,...]=name
			  
			if(rects):
				j+=1 
				print("!!!!!!!!!!!!!!!!!!j",j)
		print("i=",i)
		i+=1
		if(i>=1): break

main()

############################### after processing #####################################
#do the same with data_actresses for img[4000-8000]
#such, we will have a dataset of 8000 images containing 20 images for each person
hf_new = h5py.File(r'C:\Users\duo li\Desktop\third year\aps360\project\datacomputation\img.h5','r+')
img = hf_new.create_dataset("train_img", [520,224,224,3], dtype= np.uint8)
name = hf_new.create_dataset("names",(520,), dtype=h5py.string_dtype())
hf = h5py.File(r'C:\Users\duo li\Desktop\third year\aps360\project\datacomputation\data_actor_2.h5','r')
  
j=0
k=0
current_name=hf["names"][0].split()[0]
for i in range(4000):
	if(hf["train_img"][i,...].any()):
		if(j>=20):
			if(current_name!=hf["names"][i,...].split()[0]):
				j=0
		   
		else:
		   if(current_name!=hf["names"][i,...].split()[0]):
		   hf_new["names"][k,...]=hf["names"][i,...]
		   hf_new["train_img"][k,...]=hf["train_img"][i,...]
		   k=k+1

		   j+=1
		current_name=hf["names"][i,...].split()[0]
		
	print("i",i)
	if(k>=4000): 
		break
hf.close()
hf_new.close()
	
	
