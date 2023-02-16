"""
2022-08-09

threading
https://gmlwjd9405.github.io/2018/09/14/process-vs-thread.html
"""
import numpy as np
from keras.models import load_model
import datetime
from user_functions.influx_write import write_tag, get_ifdb
from user_functions.img_roi import pre_CAM01, pre_CAM02
import time
import cv2
from threading import Thread
import tensorflow as tf




# GPU 메모리 증가 - 다중작업에 필요
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # 프로그램 시작시에 메모리 증가가 설정되어야함
        print(e)

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
        self.pre_fn = {0: pre_CAM01, 1: pre_CAM02}[self.channel]
        self.tag_name: str = {0: "CNN_RESULT_FD1312", 1: "CNN_RESULT_FD1311"}[self.channel]
        self.url = load_dvr(n_channel=self.channel, dvr_id='rtsp' + str(channel+1))
        self.capture = None
        self.fps = 30
        self.delay: int = int(round(1000 / self.fps))
        self.model = None
        self.frame = None
        self.result = None

    def do_connect(self):
        try:
            self.capture = cv2.VideoCapture(self.url)
            # print(f"{self.channel} isOpened: {self.capture.isOpened()}")
        except:
            time.sleep(1)
            self.do_connect()

    def check_connect(self):  # 10초마다 연결 상태 확인
        while True:
            time.sleep(10)
            try:
                if self.result == -1 or not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(self.url)
                # if not self.capture.isOpened():
                #     self.capture = cv2.VideoCapture(self.url)
            except Exception as e:
                print('[check_connect error]', e)

    def reader(self):
        while True:
            try:
                self.capture.grab()
                ret, self.frame = self.capture.retrieve()
                time.sleep(self.delay / 1000 / 2)
            except Exception as e:
                self.frame = None
                print('[reader error]', e)

    def estimate(self):
        while True:
            time.sleep(3)  # 3초간격으로
            try:
                time_now = datetime.datetime.now()

                img = self.frame.copy()
                with img:
                    self.result = do_estimate(img, self.model, self.pre_fn)
                write_tag(ifdb, dt=time_now, tag_name=self.tag_name, v=self.result)
            except Exception as e:
                print('[estimate error]', e)


if __name__ == "__main__":
    print("시작: ", datetime.datetime.now())
    ifdb = get_ifdb(db="IO_INFOTROL", host="10.210.1.12")
    print("influxdb와 연결")

    CAM01 = ThreadedCamera(channel=0)
    CAM01.do_connect()
    CAM01.model = load_model("model/CAM01_MobileNet_0802.h5")
    print("CAM01 initialize")

    CAM02 = ThreadedCamera(channel=1)
    CAM02.do_connect()
    CAM02.model = load_model("model/CAM02_0808_ALL_MOBILENET.h5")
    print("CAM02 initialize")

    th1 = Thread(target=CAM01.reader)
    th2 = Thread(target=CAM01.estimate)
    th3 = Thread(target=CAM01.check_connect)
    th4 = Thread(target=CAM02.reader)
    th5 = Thread(target=CAM02.estimate)
    th6 = Thread(target=CAM02.check_connect)

    Threads = [th1, th2, th3, th4, th5, th6]
    for t in Threads:
        t.start()
    for t in Threads:
        t.join()
