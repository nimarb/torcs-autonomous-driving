#!/usr/bin/env python
import rospy
import pickle
import argparse
from torcs_msgs.msg import TORCSSensors, TORCSCtrl
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan, Image


class DataCollector:
    def __init__(self):
        self.path = 'raw_data/'
        self.data_list = []
        self.data_obj = {'sensor': None, 'ctrl': None, 'speed': None, 'laser': None, 'ctrl_state': None}

        # In ROS, nodes are uniquely named. If two nodes with the same
        # node are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
        rospy.init_node('torcs_data_collector')

        # rospy.Subscriber("/torcs_ros/pov_image", Image, self.append_data)
        rospy.Subscriber("/torcs_ros/ctrl_cmd", TORCSCtrl, self.save_ctrl_cmd)
        rospy.Subscriber("/torcs_ros/sensors_state", TORCSSensors, self.save_sensor)
        rospy.Subscriber("/torcs_ros/speed", TwistStamped, self.save_speed)
        rospy.Subscriber("/torcs_ros/scan_track", LaserScan, self.save_laser)
        rospy.Subscriber("/torcs_ros/ctrl_state", TORCSCtrl, self.save_ctrl_state)

    def save_sensor(self, data):
        # _data = {'angle': data.angle, 'displacement': data.trackPos}
        self.data_obj['sensor'] = data

    def save_ctrl_cmd(self, data):
        # _data = {'accel': data.accel, 'steering': data.steering, 'brake': data.brake}
        self.data_obj['ctrl'] = data

    def save_ctrl_state(self, data):
        # _data = {'accel': data.accel, 'steering': data.steering, 'brake': data.brake}
        self.data_obj['ctrl_state'] = data

    def save_speed(self, data):
        # _data = {'x': data.twist.linear.x, 'y': data.twist.linear.y, 'z': data.twist.linear.z}
        self.data_obj['speed'] = data

    def save_laser(self, data):
        # ranges is a list
        # _data = {'ranges': data.ranges}
        self.data_obj['laser'] = data
        self.append_data()

    def append_data(self):
        #copy dictionary
        rospy.loginfo('appended data')
        self.data_list.append(self.data_obj.copy())

    def save_img(self, data):
        pass

    def store_data(self, map_name):
        path = self.path + map_name
        with open(path, 'w') as f:
            pickle.dump(self.data_list, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='collect data from subscribing to topics')
    # parser.add_argument('backend', metavar='backend', type=str, nargs='?', default='nengo', help='')
    parser.add_argument('map_name')
    args = parser.parse_args()
    map_name = args.map_name

    print('started.. map name: ' + map_name)
    collector = DataCollector()
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    collector.store_data(map_name)
    print('exit gracefully')
