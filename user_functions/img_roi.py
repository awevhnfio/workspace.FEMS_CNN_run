"""
image ROI (관심영역) 설정
"""
import cv2
import numpy as np


def roi(image, start=(500,100), roi_size=(480,480)):
    (x, y), (w,h) = start, roi_size
    roi_img = image[y: y+h, x: x+w]
    return roi_img


def roi_v2(frame, start, roi_size):
    (x, y), (w, h) = start, roi_size
    src = frame[y: y + h, x: x + w]
    img_blur = cv2.blur(src, ksize=(4,4), anchor=None, borderType=None)
    img_gap = src - img_blur
    img_sharp = src + img_gap
    return contra2(img_sharp)


def contra2(src):
    alpha = 0.6
    dst = np.clip((1 + alpha) * src - 128 * alpha, 0, 255).astype(np.uint8)
    return dst


def pre_CAM01(frame):
    roi_image = roi(frame, start=(340, 0), roi_size=(380, 380))
    prepro_image = cv2.resize(roi_image, dsize=(224, 224), interpolation=cv2.INTER_AREA) / 255
    preprocessed_frame = prepro_image.reshape(1, 224, 224, 3)
    return preprocessed_frame


def pre_CAM02(frame):
    roi_image = roi_v2(frame, start=(520, 480-224), roi_size=(224, 224))
    prepro_image = cv2.resize(roi_image, dsize=(224, 224), interpolation=cv2.INTER_AREA) / 255
    preprocessed_frame = prepro_image.reshape(1, 224, 224, 3)
    return preprocessed_frame