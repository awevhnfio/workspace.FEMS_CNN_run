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


class ThreadedCamera:
    def __init__(self, channel: int):
        self.channel: int = channel
        self.capture = None
        self.frame = None
        self.status = False

    # dvr과 최초 연결
    def do_connect(self):
        dvr_id = 'rtsp' + str(self.channel + 1)
        dvr_pw = 'Qsc123!@'
        ip_address = '10.210.1.82'
        url = f'rtsp://{dvr_id}:{dvr_pw}@{ip_address}:558/LiveChannel/{self.channel}/media.smp/profile=3'
        while not self.status:
            try:
                self.capture = cv2.VideoCapture(url)
                self.status = self.capture.grab()  # connect 받아오기를 실패했을 경우 self.status = False
                if self.status is False:
                    self.capture.release()
                    raise ConnectionError
            except ConnectionError:
                self.status = False
                time.sleep(5)

    # 10초마다 dvr과 연결 상태 확인
    def check_connect(self):  # 10초마다 연결 상태 확인
        while True:
            time.sleep(10)
            try:
                if not self.capture.isOpened():
                    self.status = False
                if not self.status:
                    self.capture.release()
                    self.do_connect()
            except Exception as error_check_connect:
                self.status = False

    # 프레임 읽어오기
    def reader(self):
        while True:
            try:
                self.capture.grab()
                ret, self.frame = self.capture.retrieve()  # 멀티로 할 때 grab하고 retrieve해야 안 깨짐
                if (not ret) or (self.frame is None):
                    raise ConnectionError
            except ConnectionError:
                self.status = False
                self.frame = None

    # 받아온 프레임을 모델을 통해 estimate
    def estimate(self):
        # 초기 값들 불러오기
        start_size = {0: (340,0), 1: (110, 480 - 330)}[self.channel]
        roi_size = {0: (380,380), 1: (330, 330)}[self.channel]
        tag_name: str = {0: "CNN_RESULT_FD1312", 1: "CNN_RESULT_FD1311"}[self.channel]
        ifdb = get_ifdb(db="IO_INFOTROL", host="10.210.1.12")
        model_name: str = {0: "CAM01_MobileNet_0802.h5", 1: "CAM02_0808_ALL_MOBILENET.h5"}[self.channel]
        model = load_model("model/" + model_name)
        while True:
            time_now = datetime.datetime.now()
            try:
                raw_frame = self.frame.copy()
                res = do_estimate(raw_frame, model, start_size, roi_size) if self.status else -1
                if self.status:
                    self.status = False
            except:  # frame 이 None 인 경우 ConnectionError
                res = -1
                self.status = False
            write_tag(ifdb, dt=time_now, tag_name=tag_name, v=res)  # influxdb에 write
            time.sleep(5)  # 5초 간격으로 estimate


# 전처리 및 estimate
def do_estimate(raw_frame, model, start_size, roi_size):
    # 전처리
    roi_image = roi_v2(raw_frame, start=start_size, roi_size=roi_size)
    prepro_image = cv2.resize(roi_image, dsize=(224, 224), interpolation=cv2.INTER_AREA) / 255
    preprocessed_frame = prepro_image.reshape(1, 224, 224, 3)
    # estimate 후 예측한 class 번호를 res 에 할당
    estimate_class = int(np.argmax(model.predict(preprocessed_frame, verbose=0)))
    return estimate_class


if __name__ == "__main__":
    print("시작: ", datetime.datetime.now())

    CAM01 = ThreadedCamera(channel=0)
    CAM01.do_connect()
    print("CAM01 initialize")

    CAM02 = ThreadedCamera(channel=1)
    CAM02.do_connect()
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
