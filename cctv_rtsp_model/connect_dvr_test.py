"""
2022-08-09

threading
https://gmlwjd9405.github.io/2018/09/14/process-vs-thread.html
"""
import numpy as np
from keras.models import load_model
import datetime
from user_functions.influx_write import write_tag, get_ifdb
from user_functions.img_roi import roi_v2
import time
import cv2
from threading import Thread
import tensorflow as tf


# GPU 메모리 증가 - 다중작업에 필요
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         # 프로그램 시작시에 메모리 증가가 설정되어야함
#         print(e)

# 시드 고정
np.random.seed(3)
tf.compat.v1.set_random_seed(3)


def do_estimate(buffer_frame, model, pre_fn):
    if buffer_frame is None:
        return -1
    else:
        try:
            frame = buffer_frame.copy()
            preprocessed_frame = pre_fn(frame)
            result = int(np.argmax(model.predict(preprocessed_frame, verbose=0)))
            return result
        except Exception as e:
            print(e)
            return -1


def load_dvr(n_channel, dvr_id, dvr_pw='Qsc123!@', ip_address='10.210.1.82'):
    url = f'rtsp://{dvr_id}:{dvr_pw}@{ip_address}:558/LiveChannel/{n_channel}/media.smp/profile=3'
    return url


class ThreadedCamera:
    def __init__(self, channel: int):
        self.channel: int = channel
        self.tag_name: str = {0: "CNN_RESULT_FD1312", 1: "CNN_RESULT_FD1311"}[self.channel]
        self.url = "rtsp://demo:demo@ipvmdemo.dyndns.org:5541/onvif-media/media.amp?profile=profile_1_h264&sessiontimeout=60&streamtype=unicast"
        self.capture = None
        self.fps = None
        self.delay = None
        self.model = None
        self.frame = None
        self.result = None
        self.isconnected = False

    def do_connect(self):
        while True:
            try:
                self.capture = cv2.VideoCapture(self.url)
                self.fps: float = self.capture.get(cv2.CAP_PROP_FPS)  # fps = 1 / (desired FPS)
                self.delay:int = int(round(1000/self.fps))  # 33 ms
                return
            except Exception as e:
                print(e)

    def check_connect(self):  # 10초마다 연결 상태 확인
        while True:
            time.sleep(10)
            try:
                if self.result == -1 or not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(self.url)
                # if not self.capture.isOpened():
                #     self.capture = cv2.VideoCapture(self.url)

            except Exception as e:
                print(e)

    def reader(self):
        while True:
            try:
                self.capture.grab()
                ret, self.frame = self.capture.retrieve()  # 멀티로 할 때 grab하고 retrieve해야 안 깨짐
                time.sleep(self.delay / 1000 / 2)
                self.isconnected = True
            except Exception as e:

                # self.frame = None
                self.frame = None
                self.isconnected = False
                print('[reader error]', e)

    def estimate(self):
        while True:
            time.sleep(5)  # 3초간격으로
            if self.isconnected:
                try:
                    frame = self.frame.copy()

                    roi_image = roi_v2(frame, start=(520, 480 - 224), roi_size=(224, 224))
                    prepro_image = cv2.resize(roi_image, dsize=(224, 224), interpolation=cv2.INTER_AREA) / 255
                    preprocessed_frame = prepro_image.reshape(1, 224, 224, 3)

                    result = self.model.predict(preprocessed_frame, verbose=0)
                    self.result = int(np.argmax(result))
                except Exception as e:
                    # print('[estimate error]', e)
                    self.result = -1
            else:
                self.result = -1
            print('result: ', self.result)



if __name__ == "__main__":
    print("시작: ", datetime.datetime.now())
    # ifdb = get_ifdb(db="IO_INFOTROL", host="192.168.101.109")
    # print("influxdb와 연결")

    CAM01 = ThreadedCamera(channel=0)
    CAM01.do_connect()
    # print(CAM01.delay) 33
    CAM01.model = load_model("model/CAM01_MobileNet_0802.h5")
    print("CAM01 initialize")

    # CAM02 = ThreadedCamera(channel=1)
    # CAM02.do_connect()
    # CAM02.model = load_model("model/CAM02_0808_ALL_MOBILENET.h5")
    # print("CAM02 initialize")

    th1 = Thread(target=CAM01.reader)
    th2 = Thread(target=CAM01.estimate)
    th3 = Thread(target=CAM01.check_connect)
    # th4 = Thread(target=CAM02.reader)
    # th5 = Thread(target=CAM02.estimate)
    # th6 = Thread(target=CAM02.check_connect)

    Threads = [th1, th2, th3]
    for t in Threads:
        t.start()
    for t in Threads:
        t.join()
