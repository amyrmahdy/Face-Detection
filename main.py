import cv2 
import numpy as np


webCam = cv2.VideoCapture(0)

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    success, cap = webCam.read()

    capGray = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(capGray,1.1,4)
    
    for (x,y,width,height) in faces:
        cv2.rectangle(cap,[x,y],[x+width,y+height],(255,0,0),2)

    cv2.imshow("Webcam",cap)
    cv2.waitKey(1)
