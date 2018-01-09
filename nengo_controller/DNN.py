import os

import numpy as np
import cv2
from keras.models import load_model
from cv_bridge import CvBridge
import time
#from sensor_msgs.msg import Image

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join("..", "src", "models")


class DNN:
    def __init__(self, img_w=80, img_h=60, model_name="simple-both-vals/modelslearndrive-model-09880.hd5"):
        """

        """
        self._bridge = CvBridge()
        self._model_name = model_name
        #self._model = load_model(os.path.join(MODEL_DIR, self._model_name))
        self._model = load_model("/home/nb/progs/torcs-autonomous-driving/src/models/simple-both-vals/modelslearndrive-model-09880.hd5")
        self._img_width = img_w
        self._img_height = img_h
        self._img_resize_factor = self._img_width / 640

    def propagate(self, img):
        """
        calculate one forward propagation of the DNN
        :param img: img is in raw format. sensor_msgs.msg Image
        :return:
        """
        cvimg = self._bridge.imgmsg_to_cv2(img, "bgr8")
        cvimg = cv2.resize(cvimg,
                            None,
                            fx=self._img_resize_factor,
                            fy=self._img_resize_factor)

        data = np.empty(
                (1, self._img_height, self._img_width, 3),
                dtype=object)
        data[0, :, :, :] = cvimg

        t1 = time.time()
        prediction = self._model.predict(x=data, batch_size=1)
        dt = time.time() - t1
        print("TIME TO PROP: " + str(dt))

        angle = prediction[0]
        displacement = prediction[1]
        print("Pred ang: " + str(angle) + "; dis: " + str(displacement))

        return angle, displacement
