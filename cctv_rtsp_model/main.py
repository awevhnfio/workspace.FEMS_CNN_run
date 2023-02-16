"""
2022-07-07

threading
https://gmlwjd9405.github.io/2018/09/14/process-vs-thread.html
"""
import numpy as np
from keras.models import load_model
from user_functions.influx_write import *
from user_functions.img_roi import preprocess, preprocess_test
import time
import cv2
from threading import Thread


class ThreadedCamera:
    def __init__(self, channel):
        self.channel: int = channel
        self.frame = None
        self.retval = False
        self.capture = None
        self.fps = None
        self.fps_ms = None
        self.tag_name: str = {0: "CNN_RESULT_FD1312", 1: "CNN_RESULT_FD1311"}[self.channel]
        self.roi_start: tuple = {0: (340, 0), 1: (500, 200)}[self.channel]
        self.roi_size: tuple = {0: (380, 380), 1: (280, 280)}[self.channel]
        self.model = None
        self.url = 'rtsp://rtsp.stream/pattern'
        self.frame = None

    def initialize(self, model_path):
        try:
            self.capture = cv2.VideoCapture(self.url)
            self.fps: float = 1 / self.capture.get(cv2.CAP_PROP_FPS)  # fps = 1 / (desired FPS)
            self.fps_ms: int = int(self.fps * 1000)
            self.model = load_model(model_path)
            return
        except:
            self.initialize(model_path)

    def check_connect(self):  # 1초마다 연결 상태 확인
        while True:
            if self.capture.isOpened():
                pass
            else:
                time_now = datetime.datetime.now()
                print(time_now, '연결이 끊어졌습니다. 재 연결 시도...')
                self.capture = cv2.VideoCapture(self.url)
            time.sleep(1)

    def reader(self):
        while True:
            if self.capture.isOpened():
                try:
                    self.capture.grab()
                    ret, self.frame = self.capture.retrieve()  # grab하고 retrieve해야 안 깨짐
                except:
                    self.frame = None
            else:
                self.frame = None
            # time.sleep(self.fps)  # fps 만큼 delay

    def estimate(self):  # 3초간격으로
        time.sleep(3)  # 처음 실행되고 3초 대기
        while True:
            time_now = datetime.datetime.now()
            buffer_frame = self.frame
            if buffer_frame is None:
                frame = None
            else:
                frame = buffer_frame.copy()
            result = -1
            if frame is None:  # 카메라 연결이 이상한 경우 -1 기록
                print(time_now, "frame is None")
            elif not self.capture.isOpened():  # 연결되지 않았을 때 마지막 프레임 계속 읽어오지 않도록 하기 위함
                print(time_now, "capture.is not opened")
            else:
                try:
                    test_data = preprocess_test(frame, start=self.roi_start, roi_size=self.roi_size)
                    result = int(np.argmax(self.model.predict(test_data, verbose=0)))
                    # show(frame)
                except:
                    pass
            write_tag(ifdb, dt=time_now, tag_name=self.tag_name, v=result)
            print(time_now)
            time.sleep(3)  # 1초 대기

    def reconnect(self):  # 3600초마다 reconnect
        while True:
            time.sleep(3600)
            self.capture = cv2.VideoCapture(self.url)
            print('cam reconnect.')


# def show(frame):
#     # while True:
#     if frame is None: pass
#     else:
#         try:
#             cv2.imshow('frame', frame)
#             cv2.waitKey(1)
#         except Exception as e:
#             print("error msg:", e)
#     return


if __name__ == "__main__":
    print("시작: ", datetime.datetime.now())
    ifdb = get_ifdb(db="IO_INFOTROL", host="192.168.101.195")

    CAM01 = ThreadedCamera(channel=0)
    CAM01.initialize(model_path="model/keras_model.h5")
    print("CAM01 initialize")

    th1 = Thread(target=CAM01.reader)
    th2 = Thread(target=CAM01.estimate)
    th3 = Thread(target=CAM01.reconnect)
    th5 = Thread(target=CAM01.check_connect)
    # th6 = Thread(target=show, args=CAM01.frame)
    Threads = [th1, th2, th3, th5]
    for t in Threads:
        t.start()
    for t in Threads:
        t.join()
