import argparse
import json

import os
import socket
from pickle import NONE

from custom_socket import CustomSocket
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import yaml
import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from models.common import DetectMultiBackend
from utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.augmentations import letterbox
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

#config constants
SOURCE = '0'
YOLO_WEIGHTS_PATH = WEIGHTS / 'yolov5l.pt' # model.pt path(s)
STRONG_SORT_WEIGHTS = WEIGHTS / 'osnet_x0_25_msmt17.pt' # model.pt path
CONFIG_STRONGSORT = ROOT / 'strong_sort/configs/strong_sort.yaml'
SIZE = (640,640) # inference size (height, width)
DEVICE = '' # cuda device, i.e. 0 or 0,1,2,3 or cpu
CONF_THRES=0.5  # confidence threshold
IOU_THRES=0.8  # NMS IOU threshold
MAX_DET=1000  # maximum detections per image
CLASSES=None  # filter by class: --class 0, or --class 0 2 3
AGNOSTIC_NMS=False  # class-agnostic NMS
AUGMENT=False  # augmented inference
VISUALIZE=False  # visualize features
UPDATE=False  # update all models
HALF=False  # use FP16 half-precision inference
DNN=False  # use OpenCV DNN for ONNX inference
PROJECT=ROOT / 'runs/track'  # save results to project/name
NAME='exp'  # save results to project/name
EXIST_OK=False  # existing project/name ok, do not increment
SAVE_TXT=True  # save results to *.txt
SAVE_CROP=True  # save cropped prediction boxes

import mediapipe as mp
from hand_tracking_module.hand_tracking import HandTracking
import time

mp_hands = mp.solutions.hands

@torch.no_grad() 

class ObjectTracker:
    
    def __init__(self,nr_sources = 1):
        
        # Load model
        self.device = select_device(DEVICE)
        self.model = DetectMultiBackend(YOLO_WEIGHTS_PATH, device=self.device, dnn=DNN, data=None, fp16=HALF)
        
        # initialize StrongSORT
        self.cfg = get_config()
        self.cfg.merge_from_file(CONFIG_STRONGSORT)
        
        # Create as many strong sort instances as there are video sources
        self.strongsort_list = []
        for i in range(nr_sources):
            self.strongsort_list.append(
                StrongSORT(
                    STRONG_SORT_WEIGHTS,
                    self.device,
                    max_dist=self.cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=self.cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=self.cfg.STRONGSORT.MAX_AGE,
                    n_init=self.cfg.STRONGSORT.N_INIT,
                    nn_budget=self.cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=self.cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=self.cfg.STRONGSORT.EMA_ALPHA,

                )
            )
            
        # Directories
        if not isinstance(YOLO_WEIGHTS_PATH, list):  # single yolo model
            self.exp_name = str(YOLO_WEIGHTS_PATH).rsplit('/', 1)[-1].split('.')[0]
        elif type(YOLO_WEIGHTS_PATH) is list and len(YOLO_WEIGHTS_PATH) == 1:  # single models 
            self.exp_name = YOLO_WEIGHTS_PATH[0].split(".")[0]
        else:  # multiple models after --yolo_weights
            self.exp_name = 'ensemble'
        self.exp_name = NAME if NAME is not None else self.exp_name + "_" + str(self.strong_sort_weights).split('/')[-1].split('.')[0]
        self.save_dir = increment_path(Path(PROJECT) / self.exp_name, exist_ok=EXIST_OK)  # increment run
        (self.save_dir / 'tracks' if SAVE_TXT else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    def process(self,img2,s_prev_frame):
        
        # Load model attribute
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(SIZE, s=stride)  # check image size
        
        # Dataloader
        # dataset = LoadImages(img, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
        
        outputs = [None] * nr_sources
        
        # Run tracking
        self.model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

        frame_idx = 0
        im0s = img2
        # Padded resize
        im =  letterbox(im0s, SIZE[0], stride=stride, auto=pt)[0]

        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        s = ""

        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if HALF else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = VISUALIZE
        pred = self.model(im, augment=AUGMENT, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, CLASSES, AGNOSTIC_NMS, max_det=MAX_DET)
        dt[2] += time_sync() - t3

        ##################################################
        #Put names in config/data.yaml here
        # class_names = ['banana', 'ice_tea', 'water', 'corn_flakes', 'bubble_tea', 'pockys', 'pringles', 'tuna_can', 'tonic', 'coke', 'strawberry', 'apple', 'orange', 'milk', 'peach']
        #class_names = ['tea', 'milk', 'waterbottle', 'cereal', 'cocacola', 'tonic']
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'waterbottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        use_obj = ['book', 'waterbottle', 'cup', 'fork', 'spoon', 'knife', 'apple', 'banana', 'orange', 'laptop', 'mouse', 'keyboard', 'bowl', 'cell phone', 'scissors' ]
        ##################################################
        
        # Process detection        
        for i, det in enumerate(pred):  # detections per image
            print("pred =",i,det)
            # print("shape =",len(pred))
            seen += 1
            im0 = im0s.copy()
                
            prev_frames[i] = s_prev_frame
            curr_frames[i] = im0
        
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if SAVE_CROP else im0  # for save_crop
            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if self.cfg.STRONGSORT.ECC:  # camera motion compensation
                self.strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            sol = []
            result_img = im0   
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = self.strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        obj_name = class_names[int(cls)]

                        if obj_name not in use_obj:
                            continue

                        if SAVE_TXT:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            
                            print(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                            bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                            sol.append([id,cls,obj_name,int(bbox_left),int(bbox_top),int(bbox_w),int(bbox_h)])

                        if SAVE_CROP:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = f'{id} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            result_img = im0
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                self.strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

        # prev_frames[i] = curr_frames[i]
            
        return sol,result_img,curr_frames[-1]

class WhatIsThat:

    def __init__(self):
        self.HT = HandTracking()
        self.start = time.time()

    def what_is_that(self, img, formatted_bbox):

        image = img.copy()
        image = cv2.flip(image, 1)
        image.flags.writeable = False

        # hands detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hands_results = self.HT.track(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # bbox_list = self.OD.get_bbox(image)
        # formatted_bbox = self.OD.format_bbox(bbox_list)

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
    # HOST = "192.168.8.99"
    PORT = 10002

    server = CustomSocket(HOST,PORT)
    server.startServer()

    OT = ObjectTracker()
    WIT = WhatIsThat()

    while True :
        conn, addr = server.sock.accept()
        print("Client connected from",addr)
        prev_frames = None
        OT = ObjectTracker()
        while True:
            try:
                data = server.recvMsg(conn)
                # img = np.frombuffer(data,dtype=np.uint8).reshape(480,640,3)
                img = np.frombuffer(data,dtype=np.uint8).reshape(720,1280,3)
                sol, result_img, prev_frames = OT.process(img, prev_frames)
                out = {}
                obj = []
                formatted_bbox = []
                for s in sol:
                    id, cls, classname, x, y, w, h = s
                    obj.append([id, cls, classname, x, y, w, h])
                    formatted_bbox.append([classname, (x,y,w,h), False])
                out["result"] = obj
                out["n"] = len(obj)
                # server.sendMsg(conn,json.dumps(out, indent = 4))


                with torch.no_grad():
                    # feed img to model9
                    results, frame = WIT.what_is_that(cv2.flip(img, 1), formatted_bbox)
                cv2.imshow("Result image", frame)
                cv2.waitKey(1)
                what_is_that = []
                for result in results :
                    what_is_that.append((result))
                res = {"what_is_that" : what_is_that}
                print(res)
                server.sendMsg(conn,json.dumps(res))


            except Exception as e:
                print(e)
                print("CONNECTION CLOSED")
                break

if __name__ == '__main__':
    main()
