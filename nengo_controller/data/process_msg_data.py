#!/usr/bin/env python
import rospy
import pickle
import argparse
import numpy as np
from torcs_msgs.msg import TORCSSensors, TORCSCtrl
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan, Image


def has_None_msg(sample):
    if sample['sensor'] is None or sample['speed'] is None or \
                    sample['laser'] is None or sample['ctrl'] is None:
        return True
    else:
        return False


def extract_data(sample):
    import math
    input_data = list()
    output = list()
    input_data.append(sample['sensor'].angle)
    input_data.append(sample['sensor'].trackPos)
    input_data.append(math.sqrt(sample['speed'].twist.linear.x ** 2 + sample['speed'].twist.linear.y ** 2))
    input_data.extend([s/200. for s in sample['laser'].ranges])
    output.append(sample['ctrl'].accel)
    output.append(sample['ctrl'].brake)
    output.append(sample['ctrl'].steering)
    output.append(sample['ctrl'].clutch)

    return np.hstack(input_data), np.hstack(output)


class DataProcessor:
    def __init__(self, map_name):
        self.path = 'raw_data/'
        self.obj_path = self.path + map_name
        self.data = list()
        self.input_data = None
        self.output_data = None

    def read_map_data(self):
        import pickle
        objects = []
        with (open(self.obj_path, "rb")) as openfile:
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        self.data = objects[0]

    def process_data(self):
        """
        output is of shape: (n_samples, sample_size)
        :return:
        """
        input_data = list()
        output_data = list()
        for sample in self.data:
            # filter out None entries
            if has_None_msg(sample):
                continue
            # match frequency
            # extract data
            input_sample, output_sample = extract_data(sample)
            input_data.append(input_sample)
            output_data.append(output_sample)
        # convert to numpy arrays
        self.input_data = np.vstack(input_data)
        self.output_data = np.vstack(output_data)

    def save_data(self):
        np.save(self.obj_path + '_input', self.input_data)
        np.save(self.obj_path + '_output', self.output_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='process msg collections')
    # parser.add_argument('backend', metavar='backend', type=str, nargs='?', default='nengo', help='')
    parser.add_argument('map_name')
    args = parser.parse_args()
    map_name = args.map_name

    collector = DataProcessor(map_name)
    collector.read_map_data()
    collector.process_data()
    collector.save_data()
    print('exit gracefully')
