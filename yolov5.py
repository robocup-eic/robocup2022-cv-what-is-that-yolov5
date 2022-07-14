# implement class of yolov5
import torch
import json
import os
from pathlib import Path
import socket
import sys
import cv2
import numpy as np
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, time_sync
import matplotlib.pyplot as plt
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# path
CONFIG_PATH = 'config/'
WEIGHTS_PATH = 'custom_object_weight.pt'
NAMES_PATH = CONFIG_PATH + 'coco.names'
DEVICE = "0"
CFG_PATH = CONFIG_PATH + 'yolor_p6.cfg'
IMAGE_SIZE = 640


class ObjectDetection:

    def __init__(self,
                 weights=WEIGHTS_PATH,
                 data=ROOT / 'data/data.yaml',  # dataset
                 device=DEVICE,
                 half=False,
                 hide_labels=False,
                 hide_conf=False,

                 ):
        self.device = select_device(device)

        model = DetectMultiBackend(weights, device=self.device, dnn=False, data=data, fp16=half)

        stride, names, pt = model.stride, model.names, model.pt
        self.model = model
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf
        self.names = names
        self.img_size = IMAGE_SIZE
        self.stride = stride
        self.pt = pt

    def load_classes(self, path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        # filter removes empty strings (such as last line)
        return list(filter(None, names))

    def detect(self, input_image):
        print('orignal shape', input_image.shape)
        bbox_list = []

        im = self.preprocess(input_image)
        print(im.shape)

        print("recieving image with shape {}".format(im.shape))

        dt, seen = [0.0, 0.0, 0.0], 0
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        print("Inferencing ...")
        pred = self.model(im)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,
                                   classes=None, agnostic=False, max_det=1000)
        dt[2] += time_sync() - t3
        # Process predictions
        for i, det in enumerate(pred):  # per image

            annotator = Annotator(input_image, line_width=3, example=str(self.names))
            # s += '%gx%g ' % im.shape[2:]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], input_image.shape).round()
                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (
                        self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

        return input_image

    def preprocess(self, img):
        img = letterbox(img, new_shape=self.img_size, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        return img

    def get_bbox(self, input_image):
        print('orignal shape', input_image.shape)
        bbox_list = []

        im = self.preprocess(input_image)
        print(im.shape)

        print("recieving image with shape {}".format(im.shape))

        dt, seen = [0.0, 0.0, 0.0], 0
        t1 = time_sync()
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        print("Inferencing ...")
        pred = self.model(im)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45,
                                   classes=None, agnostic=False, max_det=1000)
        dt[2] += time_sync() - t3

        # Process predictions
        for i, det in enumerate(pred):  # per image

            # s += '%gx%g ' % im.shape[2:]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], input_image.shape).round()
                # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in det:
                    temp = []
                    for ts in xyxy:
                        temp.append(ts.item())
                    bbox = list(np.array(temp).astype(int))
                    bbox.append(self.names[int(cls)])
                    bbox_list.append(bbox)
        return bbox_list

    def format_bbox(self, bbox_list):
        # return list of [obj_name, (x, y, w, h), is_pointed = False]
        format_bboxs = []
        for bbox in bbox_list:
            if bbox[4] == 'person':
                pass
            else:
                format_bboxs.append([bbox[4], tuple([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]), False])
        return format_bboxs


def main():

    # HOST = socket.gethostname()
    # PORT = 10001

    OD = ObjectDetection()
    img = cv2.imread(r'D:\coding\python\pythonProject\robocup2022-cv-yolov5\data\images\300195.jpg')

    # inference
    with torch.no_grad():
        res_img = OD.detect(img)
        bbox_list = OD.get_bbox(img)
        print(bbox_list)
    # cv2.imwrite('./test_img.png', res_img)
    # plt.show()
    # print(res_img)
    # while True:
    #     conn, addr = server.sock.accept()
    #     print("Client connected from", addr)
    #     data = server.recvMsg(conn)
    #     img = np.frombuffer(data, dtype=np.uint8).reshape(720, 1080, 3)

    #     with torch.no_grad():
    #         res = OD.detect(img)
    #         bboxs = OD.get_bbox(img)

    #     counter = 0

    #     mybboxs = []

    #     for x, y, w, h, name in bboxs:
    #         temp = {}
    #         temp["x"] = int(x)
    #         temp["y"] = int(y)
    #         temp["w"] = int(w)
    #         temp["h"] = int(h)
    #         temp["class"] = name
    #         mybboxs.append(temp)

    #     result = {"bboxs": mybboxs}

    #     # print(result)

    #     server.sendMsg(conn, json.dumps(result))


if __name__ == '__main__':
    main()
