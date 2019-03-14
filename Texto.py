# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 16:52:30 2019

@author: ASUS
"""

import cv2

### load input image and convert it to grayscale
img = cv2.imread("Captura2.jpg")
#img = cv2.medianBlur(img,3) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 240, 255, 0)



#### extract all contours
_, contours, _  = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

bb_list = []
for c in contours:  
    bb = cv2.boundingRect(c)
    # save all boxes except the one that has the exact dimensions of the image (x, y, width, height)
    if (bb[0] == 0 and bb[1] == 0 and bb[2] == img.shape[1] and bb[3] == img.shape[0]):
        continue
    bb_list.append(bb)
    
   
bb_list.sort(key=lambda x:x[0])
crop_images=[]
resize_crop_images=[]

img_boxes = img.copy()
for bb in bb_list:
   x,y,w,h = bb
   cv2.rectangle(img_boxes, (x, y), (x+w, y+h), (0, 0, 255), 2)
   crop_images.append(thresh[y:y+h, x:x+w])
   
for i in crop_images:
    if len(i)<10:
        continue
    else:
    
        resize_crop_images.append(cv2.resize(i, dsize=(15, 15), interpolation=cv2.INTER_CUBIC))

    
cv2.imwrite("boxes.jpg", img_boxes)   