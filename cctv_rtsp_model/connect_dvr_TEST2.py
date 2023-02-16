import time
import cv2
from threading import Thread

class ThreadedCamera():
    def __init__(self):
        self.channel = 0
        self.fps = 30
        self.cap = cv2.VideoCapture(self.channel, cv2.CAP_ANY)
        self.frame = None
        self.delay: int = int(round(1000 / self.fps))
        self.result = -1

    def do_connect(self):
        print('연결 상태: ' ,self.cap.isOpened())
        if not self.cap.isOpened():
            time.sleep(1)
            self.do_connect()
        return

    def reader(self):
        while True:
            try:
                self.cap.grab()
                ret, self.frame = self.cap.retrieve()
                time.sleep(self.delay / 1000 / 2)
            except Exception as e:
                print('reader', e)
                self.frame = None
                time.sleep(10)

    def estimate(self):
        while True:
            try:
                print(time.time())
                img = self.frame
                if img is None:
                    print('img is None')
                    self.result = 1
                else:
                    print('img is not None')
                    self.result = -1
            except Exception as e:
                print('estimate', e)
            time.sleep(3)

    def check_connect(self):  # 10초마다 연결 상태 확인
        while True:
            time.sleep(10)
            try:
                if self.result == -1:
                    self.capture = cv2.VideoCapture(self.channel, cv2.CAP_ANY)
                if not self.capture.isOpened():
                    self.capture = cv2.VideoCapture(self.channel, cv2.CAP_ANY)
            except Exception as e:
                print('check_connect', e)


if __name__ == "__main__":
    CAM01 = ThreadedCamera()
    CAM01.do_connect()
    print("CAM01 initialize")

    th1 = Thread(target=CAM01.reader)
    th2 = Thread(target=CAM01.estimate)
    th3 = Thread(target=CAM01.check_connect)

    Threads = [th1, th2, th3]
    for t in Threads:
        t.start()
    for t in Threads:
        t.join()
