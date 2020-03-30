import cv2
import urllib
import numpy as np
from time import sleep
import math
# vcap = cv2.VideoCapture("http://192.168.0.22")
# while(1):
#     ret, frame = vcap.read()
#     cv2.imshow('VIDEO', frame)
#     cv2.waitKey(1)

# webcam = cv2.VideoCapture("http://192.168.0.22")
cap = cv2.VideoCapture("http://192.168.0.22")
x = 1
while(cap.isOpened()):
    sleep(5000)
    frameId = cap.get()
    ret, frame = cap.read()
    if (ret != True):
        break
    filename = '/home/rodrigo/projects/object/images/bees/image' + str(int(x)) + ".jpeg";
    x += 1
    cv2.imwrite(filename, frame)
cap.release()
print("Done!")
