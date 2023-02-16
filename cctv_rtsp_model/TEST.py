import cv2


def load_dvr(n_channel, dvr_id, dvr_pw='Qsc123!@', ip_address='10.210.1.82'):
    url = f'rtsp://{dvr_id}:{dvr_pw}@{ip_address}:558/LiveChannel/{n_channel}/media.smp/profile=3'
    return url


url = load_dvr(0, 'backup1')
cap = cv2.VideoCapture(url)

print(cap.isOpened())


_, frame = cap.read()
cv2.imshow('frame', frame)
cv2.waitKey(0)

