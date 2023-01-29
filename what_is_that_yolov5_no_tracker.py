import socket
import json

import torch
import numpy as np
import mediapipe as mp
import cv2

from custom_socket import CustomSocket
from yolov5 import ObjectDetection
from hand_tracking_module.hand_tracking import HandTracking
import time


mp_hands = mp.solutions.hands

class WhatIsThat:

    def __init__(self):
        self.OD = ObjectDetection()
        self.HT = HandTracking()
        self.start = time.time()

    def what_is_that(self, img):

        image = img.copy()
        
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # hands detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_results = self.HT.track(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bbox_list = self.OD.get_bbox(image)
        formatted_bbox = self.OD.format_bbox(bbox_list)

        image.flags.writeable = True

        self.HT.read_results(image, hands_results)

        # finger_list = [(startindex, midindex, length), ...]
        finger_list = [(7, 8, 200)]

        # define solution list
        obj_list = []

        # check if there is a hand
        if self.HT.hands_results.multi_hand_landmarks:
            self.HT.draw_hand()
            self.HT.draw_hand_label()
            obj_list = self.HT.point_to(formatted_bbox, finger_list)

        self.HT.draw_boxes(formatted_bbox)

        # print(obj_list)
        # 
        # # get fps
        fps = 1 / (time.time() - self.start)
        self.start = time.time()
        cv2.putText(image, "fps: " + str(round(fps, 2)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        # cv2.imshow('result image', image)
        # cv2.waitKey(1)

        return obj_list, image


def main():
    HOST = "0.0.0.0"
    PORT = 10002

    # init model
    WID = WhatIsThat()

    server = CustomSocket(HOST, PORT)
    server.startServer()

    while True :
        conn, addr = server.sock.accept()
        print("Client connected from",addr)
        while True :
            try :
                data = server.recvMsg(conn)
                # print(data)
                img = np.frombuffer(data,dtype=np.uint8).reshape(720,1280,3)
                # cv2.imshow("frame", img)
                # cv2.waitKey(1)
                with torch.no_grad():
                    # feed img to model
                    results, frame = WID.what_is_that(img)
                cv2.imshow("Result image", frame)
                cv2.waitKey(1)
                what_is_that = []
                for result in results :
                    what_is_that.append((result))
                res = {"what_is_that" : what_is_that}
                print(res)
                server.sendMsg(conn,json.dumps(res))
                
                # cv2.waitKey(1)
            except Exception as e :
                print(e)
                print("Connection Closed")
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
