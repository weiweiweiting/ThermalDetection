 # -*- coding:utf-8 -*-
import cv2
import time
import argparse
import numpy as np
from PIL import Image
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference
import statistics

# QRcode
from pyzbar import pyzbar # QRCode
import datetime
import imutils
import cv2

# 多執行緒
import threading

# Thermal
from uvctypes import *
from queue import Queue
import platform

# 人臉特徵點
import dlib

#BUF_SIZE = 2
#q = Queue(BUF_SIZE)

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}
QRCTemp = ''

class Main:
    def __init__(self):
        # 變數設定
        self.QRC_temp = ''
        self.forehead_UPPERLEFT = (0,0) # 人臉偵測框的左上點
        self.forehead_LOWERRIGHT = (0,0) # 人臉偵測框的右下點
        
        # 讀入即時影像（從camera）
        for i in range(10): # 預設camera編號不同，用迴圈去偵測
            self.cap = cv2.VideoCapture(i)
            
            if not self.cap.isOpened(): # 影片讀取失敗
                continue
            else:
                print("開啟相機編號：", i)
                break
        
        # 影像設定
        self.height = 480
        self.width = 640
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height) # 影像高度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) # 影像寬度
#        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 影像高度
#        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 影像寬度
        print("視窗尺寸：", self.height, self.width)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # 每秒顯示幾張
        print("每秒顯示幾張：", self.fps)
#        self.fourcc = cv2.VideoWriter_fourcc(*'XVID') # 設定影片編碼方式
#        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        
        # 讀入溫度影像（從Thermal）
        BUF_SIZE = 2
        self.q = Queue(BUF_SIZE)
        PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(self.py_frame_callback)
        ctx = POINTER(uvc_context)()
        dev = POINTER(uvc_device)()
        devh = POINTER(uvc_device_handle)()
        ctrl = uvc_stream_ctrl()

        res = libuvc.uvc_init(byref(ctx), 0)
        res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
        res = libuvc.uvc_open(dev, byref(devh))
        frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16, frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval))
        res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        data = self.q.get(True, 500)
                    
        BOTH_status = True
        while BOTH_status:
            RGB_status, self.img_raw = self.cap.read()
            self.img_raw = cv2.cvtColor(self.img_raw, cv2.COLOR_BGR2RGB) # 將影像BGR轉換成RGB
            
            self.data = self.q.get(True, 500)
            if self.data is not None:
                BOTH_status = RGB_status and True
                
            if BOTH_status:
#                print("hi")
                self.DetectFace()
                self.DetectQRC()
                self.DetectThermal()
            
            cv2.imshow('Face and QRC and Thermal', self.img_raw[:, :, ::-1])
            
            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
                
        libuvc.uvc_stop_streaming(devh)
        libuvc.uvc_unref_device(dev)
        libuvc.uvc_exit(ctx)
            
    
    def py_frame_callback(self, frame, userptr):
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)
        ).reshape(
        frame.contents.height, frame.contents.width
        )

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            return

        if not self.q.full():
            self.q.put(data)
    
    def raw_to_8bit(self, data):
        cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data, 8, data)
        return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)
    
    def DisplayTemperature(self, val, loc, color):
        print(val)
        val = (val - 27315) / 100.0 # 克氏轉攝氏
        x, y = loc
        x += self.forehead_UPPERLEFT[0]
        y += self.forehead_UPPERLEFT[1]
        
        cv2.putText(self.img_raw,"{0:.1f} degC".format(val), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        
        cv2.line(self.img_raw, (x - 2, y), (x + 2, y), color, 1)
        cv2.line(self.img_raw, (x, y - 2), (x, y + 2), color, 1)
        
        # 暫時顯示溫度圖，方便校正位置
#        img = self.raw_to_8bit(self.data)
#        cv2.circle(img, (x, y), 10, color, -1)
#        cv2.imshow('Lepton Radiometry', img)
        
    def DetectThermal(self): # 偵測溫度
        self.data = cv2.resize(self.data[:,:], (int(self.width), int(self.height))) # 寬, 高
        x0, y0 = self.forehead_UPPERLEFT
        x1, y1 = self.forehead_LOWERRIGHT
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(self.data[y0:y1, x0:x1]) # 取最大最小值
        mean = np.mean(self.data[y0:y1, x0:x1]) # 取平均值
        median = np.median(self.data[y0:y1, x0:x1]) # 取中位數
        
#        x, y = self.forehead_UPPERLEFT
#        value = self.data[x][y]
        self.DisplayTemperature(maxVal, maxLoc, (255, 0, 0)) # 區域內最高溫
        self.DisplayTemperature(minVal, minLoc, (0, 0, 255)) # 區域內最低溫
        self.DisplayTemperature(mean, (0,0),(235, 180, 52)) # 區域內平均溫
        self.DisplayTemperature(median, (0,30), (192, 52, 235)) # 區域內中位數
        
        
        img = self.raw_to_8bit(self.data)
        cv2.rectangle(img, (x0, y0), (x1, y1), (192, 52, 235), 2)
        maxLocX, maxLocY = maxLoc
        maxLocX += x0
        maxLocY += y0
    
        cv2.circle(img, (maxLocX, maxLocY), 10, (192, 52, 235), -1)
        cv2.imshow('Lepton Radiometry', img)

#        self.data = cv2.resize(self.data[:,:], (640, 480))
#        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(self.data)
#        self.DisplayTemperature(minVal, minLoc, (255, 0, 0))
#        self.DisplayTemperature(maxVal, maxLoc, (0, 0, 255))
    
    def ClearTempQRC(self): # 清空暫存QRC
        print("clear QRC_temp (" + self.QRC_temp + ")")
        self.QRC_temp = ''
        
    def DetectQRC(self): # 偵測QRC
        barcodes = pyzbar.decode(self.img_raw) # 抓取QRC
        for barcode in barcodes:
            # 得到剛剛找的QRC輪廓並繪製
            (x, y, w, h) = barcode.rect
            cv2.rectangle(self.img_raw, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # QRC 的data是bytes object，如果要顯示網址，則要把他轉換成string
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            
            # 將QRC data 繪製在畫面中
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(self.img_raw, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            if self.QRC_temp != '': # 如果QRC暫存有資料
                if self.QRC_temp != barcodeData: # 如果QRC暫存和現有的不同，則替換
                    self.QRC_temp = barcodeData
                threading.Timer(5.0, self.ClearTempQRC).start() # 5秒後清空暫存，之後加入傳到伺服器後，需要更久再清空暫存
            
            else: # 如果QRC暫存無資料
                self.QRC_temp = barcodeData # 將QRC data 存入暫存
            
            # 如果QRC暫存存有東西，則清空
            if self.QRC_temp != '':
                threading.Timer(5.0, self.ClearTempQRC).start() # 5秒後清空暫存，之後加入傳到伺服器後，需要更久再清空暫存

    def DetectFace(self): # 偵測人臉
        self.inference(
        conf_thresh=0.5,
        iou_thresh=0.5,
        target_shape=(260, 260),
        draw_result=True,
        show_result=False)
        
        # 偵測額頭的點
        predictor_path = 'shape_predictor_81_face_landmarks.dat'
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        
        dets = detector(self.img_raw, 0)
        for k, d in enumerate(dets):
            shape = predictor(self.img_raw, d)
            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
#            for num in range(shape.num_parts) # 印出所有68個點:
#            forehead_points = [i for i in range(17,27)] # 額頭的點（共10+13=23個）
#            forehead_points += [i for i in range(68, 81)]
            forehead_points = [18, 25, 69, 72]
            
            x0, y0 = shape.parts()[forehead_points[2]].x, shape.parts()[forehead_points[2]].y # 左上
            x1, y1 = shape.parts()[forehead_points[1]].x, shape.parts()[forehead_points[1]].y # 右下
            
            self.forehead_UPPERLEFT = [x0, y0]
            self.forehead_LOWERRIGHT = [x1, y1]

            cv2.rectangle(self.img_raw, (x0,y0), (x1,y1), (0,255,0), 2)
#            for num in forehead_points:
#                cv2.circle(self.img_raw, (shape.parts()[num].x, shape.parts()[num].y), 3, (0,255,0), -1)
                

        
    def inference(self,
                  conf_thresh=0.5,
                  iou_thresh=0.4,
                  target_shape=(160, 160),
                  draw_result=True,
                  show_result=True
                  ):
        image = self.img_raw
        # image = np.copy(image)
        output_info = []
        height, width, _ = image.shape
        image_resized = cv2.resize(image, target_shape)
        image_np = image_resized / 255.0  # 標準化0-1
        image_exp = np.expand_dims(image_np, axis=0)
        y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

        # remove the batch dimension, for batch is always 1 for inference.
        y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                     bbox_max_scores,
                                                     conf_thresh=conf_thresh,
                                                     iou_thresh=iou_thresh,
                                                     )

        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)

            if draw_result:
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
#                self.forehead_UPPERLEFT = [xmin, ymin]
#                self.forehead_LOWERRIGHT = [xmax, ymax]
#                cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

        if show_result:
            Image.fromarray(image).show()
        return output_info


if __name__ == "__main__":
    app = Main()
#    run_on_realtime(conf_thresh=0.5)



