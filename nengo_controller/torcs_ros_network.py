import nengo
import math
from torcs_msgs.msg import TORCSSensors, TORCSCtrl
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import LaserScan, Image
import numpy as np
import rospy


class TORCSInputNode(nengo.Node):
    def __init__(self, name):
        """
        Parameters
        ----------
        name : str
            An arbitrary name for the object
        """
        neuro_dimensions = 6 # angle, disp, speed, rangefinder[3]
        direct_dimensions = 2 # gear, rpm
        dimensions = neuro_dimensions + direct_dimensions
        self.data = np.zeros((dimensions,))

        rospy.Subscriber("/torcs_ros/sensors_state", TORCSSensors, self.extract_displacement)
        rospy.Subscriber("/torcs_ros/speed", TwistStamped, self.extract_speed)
        rospy.Subscriber("/torcs_ros/scan_track", LaserScan, self.extract_laser)
        rospy.Subscriber("/torcs_ros/ctrl_state", TORCSCtrl, self.extract_ctrl)

        super(TORCSInputNode, self).__init__(label=name, output=self.tick,
                                             size_in=0, size_out=dimensions)

    def extract_displacement(self, data):
        self.data[0] = data.angle
        self.data[1] = data.trackPos
        self.data[6] = data.gear
        self.data[7] = data.rpm

    def extract_speed(self, data):
        # self.data[2] = math.sqrt(data.twist.linear.x ** 2 + data.twist.linear.y ** 2)
        self.data[2] = data.twist.linear.x

    def extract_laser(self, data):
        self.data[3:6] = np.array(data.ranges[9:12]) * 1.0/200

    def extract_ctrl(self, data):
        pass


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

        dimensions = 5  # acceleration, brake, steering

        # TODO: discuss period
        self.publishing_period = period
        self.counter = 0

        # TODO: discuss queue_size
        self.pub = rospy.Publisher("/torcs_ros/ctrl_cmd", TORCSCtrl, queue_size=10)

        super(TORCSOutputNode, self).__init__(label=name, output=self.tick,
                                              size_in=dimensions, size_out=0)

    def tick(self, t, values):
        self.counter += 1
        if self.counter >= self.publishing_period:
            self.counter = 0
            self.pub.publish(self.get_msg(values))

    def get_msg(self, values):
        ctrl = TORCSCtrl()
        ctrl.accel = values[0]
        ctrl.brake = values[1]
        # ctrl.clutch = values[2]
        ctrl.clutch = 0
        ctrl.gear = int(np.round(values[3]))
        ctrl.steering = values[4]
        ctrl.focus = 0
        ctrl.meta = 0
        return ctrl

