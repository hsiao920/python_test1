<<<<<<< HEAD
import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)

if not (cap.isOpened()):
    print("could not open video device")

while cap.isOpened() :
    _,frame=cap.read()
    
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])

    mask=cv.inRange(hsv,lower_blue,upper_blue)

    res=cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow('webcam',frame)
    cv.imshow("mask",mask)
    cv.imshow('res',res)

    k=cv.waitKey(1)&0xff
    if k==27:
        break

cap.release()
=======
import cv2 as cv
import numpy as np

cap=cv.VideoCapture(0)

if not (cap.isOpened()):
    print("could not open video device")

while cap.isOpened() :
    _,frame=cap.read()
    
    hsv=cv.cvtColor(frame,cv.COLOR_BGR2HSV)

    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])

    mask=cv.inRange(hsv,lower_blue,upper_blue)

    res=cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow('webcam',frame)
    cv.imshow("mask",mask)
    cv.imshow('res',res)

    k=cv.waitKey(1)&0xff
    if k==27:
        break

cap.release()
>>>>>>> 5279899b69b29cd56fae64d120ae7e49e7589eaf
cv.destroyAllWindows()