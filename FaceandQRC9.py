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
from pyzbar import pyzbar # QRCode
import datetime
import imutils
import cv2
import threading
from uvctypes import *
from queue import Queue
import platform
import dlib
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import hashlib
import subprocess
import json
import configparser

'''
讀入設定檔 config.ini
'''
config = configparser.ConfigParser()
# 讀取設定檔
config.read('config.ini')
# 主機連線相關設定
Device = config['host']['host_device']
# 裝置地點相關設定
Location = config['location']['location_name']
# 介面參數相關設定
Direction = config['direction']['device_direction'] # 直式或橫式
ImagePath = 'img_' + Direction # 介面圖片的資料夾來源

'''
人臉偵測模型
此模型可偵測戴口罩和不戴口罩兩類人臉
'''
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

'''
人臉特徵點偵測模型
此模型先找出人臉後，在人臉標記81個facial mark點位
'''
predictor_path = 'shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

'''
RGB相機 執行緒
- 開啟相機
- 設定相機
- 啟動整個程式運作
'''
class RGBCamera(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    set_status = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.RGB_status = False
        self.RGB_img = None
        
    def OpenCamera(self): # 開啟相機
        for i in range(10): # 預設camera編號不同，用迴圈去偵測
            self.cap = cv2.VideoCapture(i)
            if not self.cap.isOpened(): # 影片讀取失敗
                continue

            else:
                print("開啟相機編號：", i)
                return True
        print("相機開啟失敗")
        return False

    def SetCamera(self): # 設定相機
        self.height = 480
        self.width = 640
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height) # 影像高度
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width) # 影像寬度
#        print("視窗尺寸：", self.height, self.width)
#        self.fps = self.cap.get(cv2.CAP_PROP_FPS) # 每秒顯示幾張
#        print("每秒顯示幾張：", self.fps)
            
    def run(self): # 開始讀入影像（啟動整個程式運作）
        face = Face() # 人臉
        thermal_camera = ThermalCamera() # 溫度
        qrcode = QRCode() # QRCode
        
        if self.OpenCamera() and thermal_camera.OpenCamera():
            while self._run_flag:
                self.RGB_status, self.RGB_img = self.cap.read()
                Thermal_status, Thermal_data = thermal_camera.ReadCamera()

                if self.RGB_status and Thermal_status:
                    face_status, face_UPPERLEFT, face_LOWERRIGHT = face.DetectFace(self.RGB_img) # 偵測人臉
                    if face_status: # 有偵測到臉
                        forehead_UPPERLEFT, forehead_LOWERRIGHT = face.DetectForehead(self.RGB_img, face_UPPERLEFT, face_LOWERRIGHT) # 偵測額頭
#                        cv2.rectangle(self.RGB_img, face_UPPERLEFT, face_LOWERRIGHT, (0,0,255), 2) # 顯示人臉框（藍色）
                        cv2.rectangle(self.RGB_img, forehead_UPPERLEFT, forehead_LOWERRIGHT, (0,255,0), 2) # 顯示額頭框（綠色）

                        # 取額頭範圍溫度
                        forehead_temperature = face.GetThermal(Thermal_data, forehead_UPPERLEFT, forehead_LOWERRIGHT)
                        cv2.putText(self.RGB_img, str(forehead_temperature), forehead_UPPERLEFT, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 額頭溫度

                    qrc_status, QRC_UPPERLEFT, QRC_LOWERRIGHT, QRC_TEXT = qrcode.DetectQRC(self.RGB_img) # 偵測QRCode
                    if qrc_status: # 有偵測到QRC
                        cv2.rectangle(self.RGB_img, QRC_UPPERLEFT, QRC_LOWERRIGHT, (255,0,0), 2) # 顯示QRCode（紅色）
#                        cv2.putText(self.RGB_img, QRC_TEXT,QRC_UPPERLEFT, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    self.change_pixmap_signal.emit(self.RGB_img)
                    
                    if qrcode.QRC_temp != '': # 如果現在暫存QRC有東西，則讓他5秒固定清空暫存
                        threading.Timer(5.0, qrcode.ClearTempQRC).start() # 5秒後清空QRC暫存
                    else: # 如果現在暫存QRC沒東西，則讓他停止清空暫存的動作
                        threading.Timer(5.0, qrcode.ClearTempQRC).cancel()
                    
                    # 前端顯示
                    # 有人臉、有二維碼、有溫度
                    if face_status and qrc_status:
                    
                        # 避免短時間內重複傳送相同QRCode
                        if qrcode.QRC_temp == QRC_TEXT: #暫存的和現在的QRC重複，不需要重送
                            continue
                        
                        else: #暫存的和現在的QRC不同，需送至主機，並將暫存更新為現在的QRC
                            print("傳送至主機")
                            qrcode.QRC_temp = QRC_TEXT
                        
#                            tempp = 35.64 # 暫時用
                            tempp = forehead_temperature
                            RFIDid = str(QRC_TEXT.split(' ')[0]) # QRC的文字
                            source_this = "VIPLab$" + RFIDid + "/" + Device + "/" +str(tempp)
                        
                            # 上傳至主機
                            m_this = hashlib.new('md5')
                            m_this.update(source_this.encode("utf-8"))
                            md5checksum_this = m_this.hexdigest()
                            cmd = 'curl --form "d=' + Device + '" --form "c='+RFIDid+'" --form "t='+str(tempp)+'" --form "s='+ md5checksum_this+' " "https://checkin.csie.ntnu.edu.tw/index.php/C_api/a_person_post" -s'
                            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

                            result = p.stdout.readline()
                            message = json.loads(result)
                            
                            if message['status'] == 'Valid': # 有效的QRC
                                if tempp <= 37.5: # 體溫正常
                                    self.set_status.emit("OK", message['name'] + "已報到，體溫正常")
#                                    print(message['name'] + "已報到，體溫正常")
                                else: # 體溫過高
                                    self.set_status.emit("Error", message['name'] + "已報到，體溫過高")
#                                    print(message['name'] + "已報到，體溫過高")
                            else: # 無效的QRC
                                self.set_status.emit("Error", "此為無效二維碼")

            cap.release()
            thermal_camera.ExitCamera()

    def stop(self):
        self._run_flag = False
        self.wait()

'''
即時時間顯示 執行緒
'''
class TimeThread(QThread):
    show_time = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def getNowDate(self):
        date = datetime.datetime.now().date()
        return "{} 年 {} 月 {} 日".format(date.year, date.month, date.day)

    def getNowTime(self):
        time = datetime.datetime.now().time()
        return "{} : {} : {}".format("0"+str(time.hour) if time.hour < 10 else time.hour, "0"+str(time.minute) if time.minute < 10 else time.minute, "0"+str(time.second) if time.second < 10 else time.second)

    def run(self):
        while self._run_flag:
            self.show_time.emit(self.getNowDate(), self.getNowTime())
            time.sleep(1)

    def stop(self):
        self._run_flag = False
        self.wait()

'''
QRC顯示 執行緒
'''
class clearThread(QThread):
    clearStatus = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        while self._run_flag:
            time.sleep(3)
            self.clearStatus.emit("Clear", "")

    def stop(self):
        self._run_flag = False
        self.wait()
        
'''
介面
'''
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.disply_width = int(config[Direction]['display_width'])
        self.display_height = int(config[Direction]['display_height'])
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(self.disply_width, self.display_height) # 設定視窗大小
        MainWindow.setStyleSheet("background-color: rgb(129, 50, 52);") # 設定背景顏色
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.Title = QtWidgets.QLabel(self.centralwidget)
        title_locat = self.parse_tuple(config[Direction]['title'])
        self.Title.setGeometry(QtCore.QRect(*title_locat))
        font = QtGui.QFont()
        font.setFamily("Hannotate TC")
        font.setPointSize(36) # 設定標題大小
        font.setBold(True)
        font.setWeight(60) # 設定標題粗細
        self.Title.setFont(font)
        self.Title.setAutoFillBackground(False)
        self.Title.setStyleSheet("color: rgb(255, 255, 255)") # 設定標題顏色
        self.Title.setAlignment(QtCore.Qt.AlignCenter)
        self.Title.setObjectName("Title")

        self.NTNULogo = QtWidgets.QLabel(self.centralwidget)
        ntnulogo_locat = self.parse_tuple(config[Direction]['NTNULogo'])
        self.NTNULogo.setGeometry(QtCore.QRect(*ntnulogo_locat))
        self.NTNULogo.setText("")
        self.NTNULogo.setPixmap(QtGui.QPixmap(ImagePath + "/NB_NTNU_logo.png")) # 設定引入師大 logo 圖片
        self.NTNULogo.setObjectName("NTNULogo")

#        self.cap = cv2.VideoCapture(0) # 讀入即時影像 (從camera)
        self.Video = QtWidgets.QLabel(self.centralwidget)
        video_locat = self.parse_tuple(config[Direction]['Video'])
        self.Video.setGeometry(QtCore.QRect(*video_locat))
        self.Video.setObjectName("Video")
        self.Video.setAlignment(QtCore.Qt.AlignCenter)

        self.Status = QtWidgets.QLabel(self.centralwidget)
        status_locat = self.parse_tuple(config[Direction]['Status'])
        self.Status.setGeometry(QtCore.QRect(*status_locat))
        self.Status.setStyleSheet("background-color: transparent;")
        self.Status.setText("")
        self.Status.setObjectName("Status")
        self.StatusText = QtWidgets.QLabel(self.centralwidget)
        statustext_locat = self.parse_tuple(config[Direction]['StatusText'])
        self.StatusText.setGeometry(QtCore.QRect(*statustext_locat))
        font = QtGui.QFont()
        font.setPointSize(14) # 設定狀態框文字大小
        font.setBold(True)
        font.setWeight(70) # 設定狀態框文字粗細
        self.StatusText.setFont(font)
        self.StatusText.setStyleSheet("background-color:transparent;")
        self.StatusText.setAlignment(QtCore.Qt.AlignCenter)
        self.StatusText.setObjectName("StatusText")

        self.DateTimeArea = QtWidgets.QLabel(self.centralwidget)
        datetimearea_locat = self.parse_tuple(config[Direction]['DateTimeArea'])
        self.DateTimeArea.setGeometry(QtCore.QRect(*datetimearea_locat))
        self.DateTimeArea.setStyleSheet("background-color: transparent;")
        self.DateTimeArea.setText("")
        self.DateTimeArea.setPixmap(QtGui.QPixmap(ImagePath + "/area.png")) # 設定引入日期時間底框圖片
        self.DateTimeArea.setObjectName("DateTimeArea")

        self.DateText = QtWidgets.QLabel(self.centralwidget)
        datetext_locat = self.parse_tuple(config[Direction]['DateText'])
        self.DateText.setGeometry(QtCore.QRect(*datetext_locat))
        font = QtGui.QFont()
        font.setFamily("Hannotate TC")
        font.setPointSize(16) # 設定日期文字大小
        font.setBold(True)
        font.setWeight(75) # 設定日期文字粗細
        self.DateText.setFont(font)
        self.DateText.setStyleSheet("background-color: transparent; color:white;") # 設定日期文字顏色
        self.DateText.setAlignment(QtCore.Qt.AlignCenter)
        self.DateText.setObjectName("DateText")

        self.TimeText = QtWidgets.QLabel(self.centralwidget)
        timetext_locat = self.parse_tuple(config[Direction]['TimeText'])
        self.TimeText.setGeometry(QtCore.QRect(*timetext_locat))
        font = QtGui.QFont()
        font.setFamily("Hannotate TC")
        font.setPointSize(16) # 設定時間文字大小
        font.setBold(True)
        font.setWeight(70) # 設定時間文字粗細
        self.TimeText.setFont(font)
        self.TimeText.setStyleSheet("background-color: transparent; color:white;") # 設定時間文字顏色
        self.TimeText.setAlignment(QtCore.Qt.AlignCenter)
        self.TimeText.setObjectName("TimeText")

        self.Location = QtWidgets.QLabel(self.centralwidget)
        location_locat = self.parse_tuple(config[Direction]['Location'])
        self.Location.setGeometry(QtCore.QRect(*location_locat))
        font = QtGui.QFont()
        font.setFamily("Hannotate TC")
        font.setPointSize(12) # 設定地點文字大小
        font.setBold(True)
        font.setWeight(70) # 設定地點文字粗細
        self.Location.setFont(font)
        self.Location.setStyleSheet("color:white") # 設定地點文字顏色
        self.Location.setObjectName("Location")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def parse_tuple(self, input): # string轉換成tuple
        return tuple(map(int, input.split(',')))
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.Title.setText(_translate("MainWindow", "智慧額溫 2.0")) # 設定標題文字
        
        # 設定地點文字
        self.Location.setText(_translate("MainWindow", Location))

        # 設定即時影像 （RGB相機 執行緒）
        self.rgb_camera = RGBCamera()
        self.rgb_camera.start()
        self.rgb_camera.change_pixmap_signal.connect(self.update_image)
        self.rgb_camera.set_status.connect(self.update_status)
        
        # 設定日期與時間文字 （即時時間顯示 執行緒）
        self.thread_2 = TimeThread()
        self.thread_2.show_time.connect(self.update_time)
        self.thread_2.start()

        # 設定時間清除文字 （QRC顯示 執行緒）
        self.thread_3 = clearThread()
        self.thread_3.clearStatus.connect(self.update_status)
        self.thread_3.start()
        
    def update_image(self, RGB_img):
        qt_img = self.convert_cv_qt(RGB_img)
        self.Video.setPixmap(qt_img)
        
    def update_time(self, date, time):
        self.DateText.setText(date)
        self.TimeText.setText(time)

    def update_status(self, status, text):
        if status == "Error": # Error Status
            self.Status.setPixmap(QtGui.QPixmap(ImagePath + "/NB_QRC_Error.png"))
        elif status == "Clear": # Clear Status
            self.Status.setPixmap(QtGui.QPixmap(""))
        else: # OK Status
            self.Status.setPixmap(QtGui.QPixmap(ImagePath + "/NB_QRC_OK.png"))
        self.StatusText.setText(text) # Status Text
        
        
    def convert_cv_qt(self, RGB_img):
        rgb_image = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
#        rgb_image = cv2.flip(rgb_image, 1) # 鏡像翻轉
#        rgb_image = cv2.rotate(rgb_image, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE) # 順時針旋轉90度
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        if Direction == 'horizontal':
            p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio) # 設定影像大小
        elif Direction == 'vertical':
            p = convert_to_Qt_format.scaled(600, 600, Qt.KeepAspectRatioByExpanding) # 設定影像大小
        return QtGui.QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

'''
溫度
- 開啟相機
- 讀取相機
- 離開相機
'''
class ThermalCamera:
    def __init__(self):
        BUF_SIZE = 2
        self.q = Queue(BUF_SIZE)
        self.PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(self.py_frame_callback)

    def OpenCamera(self): # 開啟相機
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        ctrl = uvc_stream_ctrl()

        res = libuvc.uvc_init(byref(self.ctx), 0)
        if res < 0:
            print("uvc_init error")
            exit(1)

        res = libuvc.uvc_find_device(self.ctx, byref(self.dev), PT_USB_VID, PT_USB_PID, 0)
        if res < 0:
            print("uvc_find_device error")
            exit(1)

        res = libuvc.uvc_open(self.dev, byref(self.devh)) # 開啟裝置
        if res < 0:
            print("uvc_open error")
            exit(1)

        print("device opened!")

        print_device_info(self.devh)
        print_device_formats(self.devh)

        frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
        if len(frame_formats) == 0:
            print("device does not support Y16")
            exit(1)

        libuvc.uvc_get_stream_ctrl_format_size(self.devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
          frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
        )

        res = libuvc.uvc_start_streaming(self.devh, byref(ctrl), self.PTR_PY_FRAME_CALLBACK, None, 0)
        if res < 0:
            print("uvc_start_streaming failed: {0}".format(res))
            exit(1)
            return False

        else:
            self.thermal_data = self.q.get(True, 500)
            return True

    def raw_to_8bit(self, data): # 溫度資料格式轉換
        cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
        np.right_shift(data, 8, data)
        return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

    def ReadCamera(self): # 讀取相機
        self.thermal_data = self.q.get(True, 500)
        if self.thermal_data is not None:
#            img = self.raw_to_8bit(self.thermal_data)
#            cv2.imshow('Lepton Radiometry', img)
#            cv2.waitKey(1)
            return True,self.thermal_data
        else:
            return False, []

    def ExitCamera(self): # 離開相機
        libuvc.uvc_stop_streaming(self.devh)
        libuvc.uvc_unref_device(self.dev)
        libuvc.uvc_exit(self.ctx)
        cv2.destroyAllWindows()

    def py_frame_callback(self, frame, userptr):

        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.frombuffer(
          array_pointer.contents, dtype=np.dtype(np.uint16)
        ).reshape(
          frame.contents.height, frame.contents.width
        ) # no copy

        # data = np.fromiter(
        #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
        # ).reshape(
        #   frame.contents.height, frame.contents.width, 2
        # ) # copy

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
          return

        if not self.q.full():
          self.q.put(data)

'''
人臉
- 擷取額頭
- 偵測人臉
- 擷取額溫
- 溫度單位轉換
'''
class Face:
    def DetectForehead(self, image, face_UPPERLEFT, face_LOWERRIGHT): # 擷取額頭，回傳額頭左上點（upperleft）、額頭右下點（lowerright）
        # 偵測額頭的點
        face_trans = (face_LOWERRIGHT[0] - face_UPPERLEFT[0]) // 6 # x座標位移為人臉偵測框的1/6
        forehead_x0 = face_UPPERLEFT[0] + face_trans # x座標開始位置為本來x+位移
        forehead_x1 = face_LOWERRIGHT[0] - face_trans # x座標結束位置為本來x-位移
        forehead_height = (face_LOWERRIGHT[1] - face_UPPERLEFT[1]) // 4 # 長度為人臉偵測匡的1/4
        forehead_UPPERLEFT = (forehead_x0, face_UPPERLEFT[1])
        forehead_LOWERRIGHT = (forehead_x1, face_UPPERLEFT[1] + forehead_height)

#         dlib 偵測更精準的額頭位置
#        predictor_path = 'shape_predictor_81_face_landmarks.dat'
#        detector = dlib.get_frontal_face_detector()
#        predictor = dlib.shape_predictor(predictor_path)
#
#        dets = detector(image, 0)
#        for k, d in enumerate(dets):
#            shape = predictor(image, d)
#            landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
#            forehead_points = [18, 25, 69, 72]
#
#            x0, y0 = shape.parts()[forehead_points[2]].x, shape.parts()[forehead_points[2]].y # 左上
#            x1, y1 = shape.parts()[forehead_points[1]].x, shape.parts()[forehead_points[1]].y # 右下
#
#            forehead_UPPERLEFT = (x0, y0)
#            forehead_LOWERRIGHT = (x1, y1)
#
#            return forehead_UPPERLEFT, forehead_LOWERRIGHT
        
        return forehead_UPPERLEFT, forehead_LOWERRIGHT
        
    def DetectFace(self, image): # 偵測人臉，回傳是否有偵測到（status）、人臉左上點（upperleft）、人臉右下點（lowerright）
        status, upperleft, lowerright = self.inference(image, conf_thresh=0.5, iou_thresh=0.5, target_shape=(260, 260), draw_result=True, show_result=False)
        if status:
            return status, upperleft, lowerright
        else:
            return status, upperleft, lowerright
            
    def ThermalTransfer(self, temp): # 溫度轉換
        temp = (temp - 27315) / 100.0 # 克氏轉攝氏
        return temp
        
    def GetThermal(self, thermal_data, forehead_UPPERLEFT, forehead_LOWERRIGHT): # 擷取額溫，回傳溫度
        thermal_data = cv2.resize(thermal_data[:,:], (int(640), int(480))) # 寬, 高
        x0, y0 = forehead_UPPERLEFT
        x1, y1 = forehead_LOWERRIGHT
        # 取最大最小值
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data)
#        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(thermal_data[y0:y1, x0:x1])
        
#        # 取平均值
#        mean = np.mean(thermal_data[y0:y1, x0:x1]) # 取平均值
#
#        # 取中位數
#        median = np.median(thermal_data[y0:y1, x0:x1]) # 取中位數
        return self.ThermalTransfer(maxVal)
                
    def inference(self, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(160, 160), draw_result=True, show_result=True): # 人臉偵測模型函式
#        image = self.img_raw
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
        keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=conf_thresh, iou_thresh=iou_thresh,)

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
                return True, (xmin, ymin), (xmax, ymax)

#        if show_result:
#            Image.fromarray(image).show()
            
        return False, (0,0), (0,0)

'''
QRCode
- 清空暫存QRC
- 偵測QRC
'''
class QRCode:
    def __init__(self):
        self.QRC_temp = ''
        
    def ClearTempQRC(self): # 清空暫存QRC
        print("clear QRC_temp (" + self.QRC_temp + ")")
        self.QRC_temp = ''
         
    def DetectQRC(self, image): # 偵測QRC，回傳是否有偵測到、QRC位置、QRC內容
        barcodes = pyzbar.decode(image) # 抓取QRC
    
        for barcode in barcodes:
            # 得到剛剛找的QRC輪廓並繪製
            (x, y, w, h) = barcode.rect

            # QRC 的data是bytes object，如果要顯示網址，則要把他轉換成string
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            
            # 將QRC data 繪製在畫面中
            text = "{} ({})".format(barcodeData, barcodeType)
                
            return True, (x,y), (x+w,y+h), text
        
        return False, (0,0), (0,0), ""



class Main:
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow() # UI介面
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())

if __name__ == "__main__":
    app = Main()



