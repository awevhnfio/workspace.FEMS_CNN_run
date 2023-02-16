"""
2022-08-29
model.predict 함수 테스트
"""
import numpy as np
from keras.models import load_model
import cv2

model = load_model("model/CAM02_0829_MOBILENET_bsize16.h5")

cap = cv2.imread("testimg.jpg")
prepro_image = cv2.resize(cap, dsize=(224, 224), interpolation=cv2.INTER_AREA) / 255
preprocessed_frame = prepro_image.reshape(1, 224, 224, 3)
res = model.predict(preprocessed_frame)

print('', np.argmax(res))
input()