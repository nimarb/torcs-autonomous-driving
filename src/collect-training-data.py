#!/usr/bin/env python2

"""
collect-training-data.py
    parameters:
        ~
        ~
    publications: 
    services: 
"""

import rospy
import roslib; roslib.load_manifest('img_to_sensor_data')
import os, os.path
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from torcs_msgs.msg import TORCSSensors

current_dir = os.path.abspath(os.path.dirname(__file__))

class training_data_collector(object):
    """collector"""

    def __init__(self):
        self.started = False
        self.data_collection_size = 500
        self.angle_array = np.empty(self.data_collection_size)
        self.distance_array = np.empty(self.data_collection_size)
        self.counter = 0
        self.tmp_angle_array = np.empty(self.data_collection_size)
        self.tmp_distance_array = np.empty(self.data_collection_size)
        self.tmp_counter = 0
        rospy.init_node("TrainingDataCollector", anonymous=True)
        rospy.on_shutdown(self.shutdown)
        self.img_path = os.path.join(current_dir, "..", "data", "images")
        self.sensor_path = os.path.join(current_dir, "..", "data", "sensor")
        self.angle_path = os.path.join(current_dir, "..", "data", "sensor", "angle")
        self.distance_path = os.path.join(current_dir, "..", "data", "sensor", "distance")
        self.bridge = CvBridge()
        self.get_data()


    def get_data(self):
        print("Start getting data")
        self.sub_sens = rospy.Subscriber("torcs_ros/sensors_state", TORCSSensors, self.get_sensor_data_cb)
        self.sub_img = rospy.Subscriber("torcs_ros/pov_image", Image, self.get_img_data_cb)


    def get_img_data_cb(self, data):
        if self.counter == self.data_collection_size:
            print("collected all samples, done")
            np.save(self.angle_path, self.angle_array)
            np.save(self.distance_path, self.distance_array)
            return
        else:
            self.angle_array[self.counter] = self.tmp_angle_array[self.tmp_counter]
            self.distance_array[self.counter] = self.tmp_distance_array[self.tmp_counter]
            self.counter += 1

        cvimg = self.bridge.imgmsg_to_cv2(data, "bgr8")
        cvimg = cv2.resize(cvimg, None, fx=0.125, fy=0.125)
        cv2.imwrite(self.img_path + "/" + str(data.header.seq) + ".jpg", cvimg)
        print(data.header.seq)
        #cv2.startWindowThread()
        #cv2.namedWindow("image")
        #cv2.imshow("image", cvimg)
        #cv2.waitKey()


    def get_sensor_data_cb(self, data):
        # angle is in "angle", distance in "trackPos"

        self.tmp_angle_array[self.tmp_counter] = data.angle
        self.tmp_distance_array[self.tmp_counter] = data.trackPos
        self.tmp_counter += 1
        if self.tmp_counter == self.data_collection_size:
            self.tmp_counter = 0

        #print(data.header.seq)
        #print(data.angle)
        #print(data.trackPos)

    def shutdown(self):
        rospy.loginfo("Shutting down")


if __name__ == "__main__":
    start = training_data_collector()
    rospy.spin()

