#align all images from folder
from google.colab.patches import cv2_imshow
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor and the face aligner
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
fa = FaceAligner(predictor, desiredFaceWidth=256)

# load the input image, resize it, and convert it to grayscale
  image = cv2.imread(r'/content/drive/My Drive/Project/UTKFace/'+str(idx)+'.jpg')
  if(image==None):
    continue
  cv2_imshow(image)  
  image = imutils.resize(image, width=500)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  rects = detector(gray, 2)

  # loop over the face detections
  for rect in rects:
    (x, y, w, h) = rect_to_bb(rect)
  
    y=max(0, y)
    x=max(0, x)
    faceOrig = imutils.resize(image[y:y + h, x:x + w], width=244)
    faceAligned = fa.align(image, gray, rect)

    faceFinal = imutils.resize(faceAligned[60:196,60:196], width=244)
   
    print(idx)
    cv2.waitKey(0)
  cv2.imwrite('/content/drive/My Drive/Project/UTKFace_cropped_aligned/'+str(idx)+'.jpg',faceFinal)

