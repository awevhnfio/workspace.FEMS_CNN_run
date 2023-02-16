from keras.models import load_model

model_name1: str = {0: "CAM01_MobileNet_0802.h5", 1: "CAM02_0808_ALL_MOBILENET.h5"}[0]
model_name2: str = {0: "CAM01_MobileNet_0802.h5", 1: "CAM02_0808_ALL_MOBILENET.h5"}[1]
model1 = load_model("model/" + model_name1)
model2 = load_model("model/" + model_name2)

import numpy as np
frame = np.zeros((1,224,224,3))
res = model1.predict(frame, verbose=0)
frame = np.ones((1,224,224,3))
res2 = model2.predict(frame, verbose=0)


def fn1(model):
    print(id(model))
    return

fn1(model1)
