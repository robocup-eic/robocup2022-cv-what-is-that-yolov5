from yolov5 import ObjectDetection
from hand_tracking_module.hand_tracking import HandTracking
import torch
import numpy as np
import mediapipe as mp
import cv2
import time

mp_hands = mp.solutions.hands

class WhatIsThat:

    def __init__(self):
        self.OD = ObjectDetection()
        self.HT = HandTracking()
        self.start = time.time()

    def what_is_that(self, img):

        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # hands detection
        hands_results = self.HT.track(image)
        bbox_list = self.OD.get_bbox(image)
        formatted_bbox = self.OD.format_bbox(bbox_list)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        # get fps
        fps = 1 / (time.time() - self.start)
        self.start = time.time()
        cv2.putText(image, "fps: " + str(round(fps, 2)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow('result image', image)

        return obj_list


def main():

    # init model
    WID = WhatIsThat()

    cap = cv2.VideoCapture(2)




    while cap.isOpened():
	
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue


       # feed img to model
        i_see = WID.what_is_that(frame)

        print(i_see) 


        if cv2.waitKey(1) == ord("q"):
            cap.release()
 
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
