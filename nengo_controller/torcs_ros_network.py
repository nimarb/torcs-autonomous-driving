import nengo
import math
from torcs_msgs.msg import TORCSSensors, TORCSCtrl
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan, Image
import numpy as np
import os
# import torcs_client
import rospy
import nengo_ros.nengo_ros as nros


class TORCSROSNetwork(nengo.Network):
    def __init__(self, label='TORCSROSNetwork', b_init_node=True):
        super(TORCSROSNetwork, self).__init__(label=label)

        with self:
            if b_init_node:
                rospy.init_node('nengo_ros_torcs', anonymous=True, disable_signals=True)
            # Subscriber - input
            # self.torcs_ctrl_state = nros.TORCSCtrlSubNode(name='ctrl_subscriber_node', topic='torcs_ros/ctrl_state')
            self.torcs_sensors_state = nros.TORCSSensorsSubNode(name='sensors_subscriber_node',
                                                                topic='torcs_ros/sensors_state')
            self.scan_track = nros.laserScanSubNode(name='scan_track_subscriber_node', topic='torcs_ros/scan_track',
                                                    ranges_dim=19)
            self.speed = nros.TwistStampedSubNode(name='speed_subscriber_node', topic='torcs_ros/speed')
            # Publisher - output
            self.torcs_ctrl_cmd = nros.TORCSCtrlPubNode(name='ctrl_publisher_node', topic='torcs_ros/ctrl_cmd')


class TORCSInputNode(nengo.Node):
    def __init__(self, name, topic, trans_fnc):
        """
        Parameters
        ----------
        name : str
            An arbitrary name for the object
        topic : str
            The name of the ROS topic that is being subscribed to
        trans_fnc : callable
            A function that will transform the ROS message of msg_type to an array
            with the appropriate number of dimensions to be used as output
        """
        dimensions = 22  # angle, disp, speed, rangefinder[19]
        self.data = np.zeros((dimensions,))

        rospy.Subscriber("/torcs_ros/sensors_state", TORCSSensors, self.extract_displacement)
        rospy.Subscriber("/torcs_ros/speed", TwistStamped, self.extract_speed)
        rospy.Subscriber("/torcs_ros/scan_track", LaserScan, self.extract_laser)

        super(TORCSInputNode, self).__init__(label=name, output=self.tick,
                                             size_in=0, size_out=dimensions)

    def extract_displacement(self, data):
        self.data[0] = data.angle
        self.data[1] = data.trackPos

    def extract_speed(self, data):
        self.data[2] = math.sqrt(data.twist.linear.x ** 2 + data.twist.linear.y ** 2)

    def extract_laser(self, data):
        self.data[3:] = np.array(data.ranges)

    def tick(self, t):
        return self.data


class TORCSOutputNode(nengo.Node):
    def __init__(self, name, period=30):
        """
        Parameters
        ----------
        name : str
            An arbitrary name for the object
        period : int
            How many time-steps to wait before publishing a message. A value of 1
            will publish at every time step
        """

        dimensions = 3  # acceleration, brake, steering

        # TODO: discuss period
        self.publishing_period = period
        self.counter = 0

        rospy.Subscriber("/torcs_ros/ctrl_cmd", TORCSCtrl, self.save_ctrl)

        # TODO: discuss queue_size
        self.pub = rospy.Publisher("/torcs_ros/ctrl_cmd", TORCSCtrl, queue_size=10)

        super(TORCSOutputNode, self).__init__(label=name, output=self.tick,
                                              size_in=dimensions, size_out=0)

    def tick(self, t):
        self.counter += 1
        if self.counter >= self.publishing_period:
            self.counter = 0
            self.pub.publish(self.get_msg())

    def get_msg(self):
        # TODO: create msg
        pass

