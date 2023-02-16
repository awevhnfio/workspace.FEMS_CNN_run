"""
2022-09-16
model.predict 함수 테스트
"""
import numpy as np
from keras.models import load_model
import cv2
import os
from user_functions.img_roi import pre_CAM02
from user_functions.influx_write import write_tag, get_ifdb
from datetime import datetime as dt
from datetime import timedelta


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

ifdb = get_ifdb(db="IO_INFOTROL")
print('ifdb load')

# model_path = "D:\\02+fems_cnn\\모델 파일\\models"
# model_name = "CAM02_0901_MOBILENET_bsize16.h5"
# model = load_model(os.path.join(model_path, model_name))

# CAM01_MobileNet_0802.h5
path = "D:\\02+fems_cnn\\모델 파일\\models\\" + "CAM02_0901_MOBILENET_bsize16.h5"
model = load_model(path)

print('model load')

video_path = "C:\\cnn\\train_0923_video\\cam02"

fps = 30
video_list = os.listdir(video_path)

for video_name in video_list[1:]:
    print('video:', video_name)
    # CAM 02(10.210.1.82)_20220907_010000_033332
    cap = cv2.VideoCapture(os.path.join(video_path, video_name))
    start_dt = dt.strptime(video_name[:35], "CAM 02(10.210.1.82)_%Y%m%d_%H%M%S")
    count = 0

    while cap.isOpened():
        _ = cap.grab()
        if count % int(fps / 3) == 0:  # 초당 3 프레임
            _, frame = cap.retrieve()
            if frame is None:
                break
            preprocessed_frame = pre_CAM02(frame)
            time_ = start_dt + timedelta(seconds=(1 / fps) * count) + timedelta(minutes=13)
            res = np.argmax(model.predict(preprocessed_frame))
            write_tag(ifdb, dt=time_, tag_name="CNN_RESULT_CAM02_SIMUL_0930", v=res)
        count += 1

    cap.release()

# CAM02_0901_MOBILENET_bsize16.h5 -> CNN_RESULT_CAM02_SIMUL_0930
