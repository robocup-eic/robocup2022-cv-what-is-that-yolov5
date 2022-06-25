import socket
import cv2
import numpy as np
import time
from custom_socket import CustomSocket
import json

#image = cv2.imread("test_pics/test2.jpg")
#print(image.shape)

host = socket.gethostname()
port = 10002

c = CustomSocket(host,port)
c.clientConnect()

cap = cv2.VideoCapture(0)
cap.set(4,720)
cap.set(3,1280)

while cap.isOpened():
	
    ret, frame = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue
    
    cv2.imshow('test', frame)

    #print("Send")
    msg = c.req(frame)
    print("                                               {}".format(msg))
    #time.sleep(1)

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()





#c = CustomSocket(host,port)
#c.clientConnect()
# print(image.tobytes())
#while True : 
#    print("Send")
#    msg = c.req(cap)
#    print(msg)
#    time.sleep(10)

