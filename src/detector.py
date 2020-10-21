# -*- coding: utf-8 -*-
import cv2
from keras.models import load_model
import numpy


model_path = "../trained_model/asl-sign_recod_10_acc-0.978276.model"

labels = [
    "A",
    "B",
    "C",
    "D",
    "del",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "nothing",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "space",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

_model = load_model(model_path)

model_input_size = _model.input_shape[1:3]
cap = cv2.VideoCapture(0)

x_mid = int(cap.get(3) / 2)
y_mid = int(cap.get(4) / 2)

while True:
    ret_val, frame = cap.read()
    if ret_val == True:
        key = ''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detection_region = gray[
            (y_mid - 200) : (y_mid + 200), (x_mid - 200) : (x_mid + 200)
        ]
        detection_region = cv2.resize(
            detection_region, model_input_size, interpolation=cv2.INTER_CUBIC
        )
        preprocessed_img = detection_region.astype("float32")
        preprocessed_img /= 255
        expanded_dimen_img = numpy.expand_dims(preprocessed_img, 0)
        expanded_dimen_img = numpy.expand_dims(expanded_dimen_img, -1)

        probabilities = _model.predict(expanded_dimen_img)
        max_prob = numpy.max(probabilities)
        label = numpy.argmax(probabilities)

        if cv2.waitKey(1) & 0xFF == ord("d"):
            key = labels[label]

        cv2.rectangle(
            frame,
            ((x_mid - 200), (y_mid - 200)),
            ((x_mid + 200), (y_mid + 200)),
            (0, 0, 255),q
            3,
        )
        cv2.putText(
            frame,
            key,
            ((x_mid - 200), (y_mid - 200)),
            cv2.FONT_HERSHEY_COMPLEX,
            2,
            (0, 255, 0),
            5,
        )
        cv2.imshow("recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
cap.release()
