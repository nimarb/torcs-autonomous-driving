"""
Common templates for constructing nodes that can communicate through ROS 
(Robot Operating System)
"""

import nengo
import rospy
import tf
import numpy as np

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Wrench, Quaternion, Twist, PoseStamped
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Twist, TwistStamped
from sensor_msgs.msg import LaserScan
from torcs_msgs.msg import *
from sensor_msgs.msg import Image
import json


class RosPubNode(nengo.Node):
    def __init__(self, name, topic, dimensions, msg_type, trans_fnc, period=30):
        """
        Parameters
        ----------
        name : str
            An arbitrary name for the object
        topic : str
            The name of the ROS topic that is being published to
        dimensions : int
            The number of input dimensions that this node will accept
        msg_type : msg
            The type of ROS message that will be published
        trans_fnc : callable
            A function that will transform the input into a valid ROS message of
            msg_type
        period : int
            How many time-steps to wait before publishing a message. A value of 1
            will publish at every time step
        """
        self.publishing_period = period
        self.counter = 0
        self.trans_fnc = trans_fnc
        self.msg_type = msg_type
        self.dimensions = dimensions
        self.topic = topic

        self.pub = rospy.Publisher(topic, msg_type, queue_size=10)

        super(RosPubNode, self).__init__(label=name, output=self.tick,
                                         size_in=dimensions, size_out=0)

    def tick(self, t, values):
        self.counter += 1
        if self.counter >= self.publishing_period:
            self.counter = 0
            msg = self.trans_fnc(values)
            self.pub.publish(msg)


class RosSubNode(nengo.Node):
    def __init__(self, name, topic, dimensions, msg_type, trans_fnc):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    dimensions : int
        The number of dimensions that this node will output
    msg_type : msg
        The type of ROS message that is being subscribed to
    trans_fnc : callable
        A function that will transform the ROS message of msg_type to an array
        with the appropriate number of dimensions to be used as output
    """
        self.trans_fnc = trans_fnc
        self.msg_type = msg_type
        self.dimensions = dimensions
        self.topic = topic

        self.rval = [0] * dimensions

        self.sub = rospy.Subscriber(topic, msg_type, self.callback)

        super(RosSubNode, self).__init__(label=name, output=self.tick,
                                         size_in=0, size_out=dimensions)

    def callback(self, data):
        self.rval = self.trans_fnc(data)

    def tick(self, t):
        return self.rval


class TORCSCtrlSubNode(RosSubNode):
    def __init__(self, name, topic, mask=[1, 1, 1, 1, 1]):

        self.mask = mask
        self.dimensions = mask.count(True)

        def fn(data):
            rval = [0] * self.dimensions

            index = 0

            if self.mask[0]:
                rval[index] = data.accel
                index += 1
            if self.mask[1]:
                rval[index] = data.brake
                index += 1
            if self.mask[2]:
                rval[index] = data.clutch
                index += 1
            if self.mask[3]:
                rval[index] = data.gear
                index += 1
            if self.mask[4]:
                rval[index] = data.steering
                index += 1

            return rval

        self.fn = fn

        super(TORCSCtrlSubNode, self).__init__(name=name, topic=topic,
                                               dimensions=self.dimensions,
                                               msg_type=TORCSCtrl, trans_fnc=self.fn)


class TORCSSensorsSubNode(RosSubNode):
    def __init__(self, name, topic, mask=[1, 1, 1, 1]):

        self.mask = mask
        self.dimensions = mask.count(True)

        def fn(data):
            rval = [0] * self.dimensions

            index = 0

            if self.mask[0]:
                rval[index] = data.angle
                index += 1
            if self.mask[1]:
                rval[index] = data.gear
                index += 1
            if self.mask[2]:
                rval[index] = data.rpm
                index += 1
            if self.mask[3]:
                rval[index] = data.trackPos
                index += 1

            return rval

        self.fn = fn

        super(TORCSSensorsSubNode, self).__init__(name=name, topic=topic,
                                                  dimensions=self.dimensions,
                                                  msg_type=TORCSSensors, trans_fnc=self.fn)


class laserScanSubNode(RosSubNode):
    def __init__(self, name, topic, ranges_dim):
        self.dimensions = ranges_dim

        def fn(data):
            rval = data.ranges

            return rval

        self.fn = fn

        super(laserScanSubNode, self).__init__(name=name, topic=topic,
                                               dimensions=self.dimensions,
                                               msg_type=LaserScan, trans_fnc=self.fn)


class TwistStampedSubNode(RosSubNode):
    """
  This node subscribes to forward and angular velocity
  """

    def __init__(self, name, topic, mask=[1, 1, 1, 1, 1, 1]):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published
    mask : list
        List of boolean representing which dimensions of the Twist message
        are being subscribed to. All others will be ignored.
    """

        self.mask = mask
        self.dimensions = mask.count(True)

        def fn(data):

            rval = [0] * self.dimensions
            index = 0

            if self.mask[0]:
                rval[index] = data.twist.linear.x
                index += 1
            if self.mask[1]:
                rval[index] = data.twist.linear.y
                index += 1
            if self.mask[2]:
                rval[index] = data.twist.linear.z
                index += 1
            if self.mask[3]:
                rval[index] = data.twist.angular.x
                index += 1
            if self.mask[4]:
                rval[index] = data.twist.angular.y
                index += 1
            if self.mask[5]:
                rval[index] = data.twist.angular.z
                index += 1

            return rval

        self.fn = fn

        super(TwistStampedSubNode, self).__init__(name=name, topic=topic,
                                                  dimensions=self.dimensions,
                                                  msg_type=TwistStamped, trans_fnc=self.fn)


class TORCSCtrlPubNode(RosPubNode):
    def __init__(self, name, topic):
        self.dimensions = 5

        def fn(values):
            ctrl = TORCSCtrl()
            ctrl.accel = values[0]
            ctrl.brake = values[1]
            ctrl.clutch = values[2]
            ctrl.gear = int(np.round(values[3]))
            ctrl.steering = values[4]
            ctrl.focus = 0
            ctrl.meta = 0
            return ctrl

        self.fn = fn

        super(TORCSCtrlPubNode, self).__init__(name=name, topic=topic,
                                               dimensions=self.dimensions,
                                               msg_type=TORCSCtrl, trans_fnc=self.fn)


class FloatPubNode(RosPubNode):
    def __init__(self, name, topic):
        self.dimensions = 1

        def fn(values):
            ros_float = Float32()
            ros_float.data = values[0]

            return ros_float

        self.fn = fn

        super(FloatPubNode, self).__init__(name=name, topic=topic,
                                           dimensions=self.dimensions,
                                           msg_type=Float32, trans_fnc=self.fn)


class ForceTorqueNode(RosPubNode):
    """
  This node publishes a force and torque
  """

    def __init__(self, name, topic,
                 mask=[True, True, True, True, True, True]):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published
    mask : list
        List of boolean representing the which dimensions of the Wrench message
        are being published. All others will be left as zero.
        For Example, [True, False, False, False, False, True] will publish the
        the force in the x direction, as well as the torque in the z direction.
        Since only two fields of the message are being used, the node will have
        only two dimensions. The format is:
          [force.x, force.y, force.z, torque.x, torque.y, torque.z]
    """

        self.mask = mask
        self.dimensions = mask.count(True)

        def fn(values):
            wrench = Wrench()
            index = 0
            if self.mask[0]:
                wrench.force.x = values[index]
                index += 1
            if self.mask[1]:
                wrench.force.y = values[index]
                index += 1
            if self.mask[2]:
                wrench.force.z = values[index]
                index += 1
            if self.mask[3]:
                wrench.torque.x = values[index]
                index += 1
            if self.mask[4]:
                wrench.torque.y = values[index]
                index += 1
            if self.mask[5]:
                wrench.torque.z = values[index]
                index += 1
            return wrench

        self.fn = fn

        super(ForceTorqueNode, self).__init__(name=name, topic=topic,
                                              dimensions=self.dimensions,
                                              msg_type=Wrench, trans_fnc=self.fn)


class TwistNode(RosPubNode):
    """
  This node publishes forward and angular velocity
  """

    def __init__(self, name, topic):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published
    """

        self.dimensions = 6

        def fn(values):
            twist = Twist()
            twist.linear.x = values[0]
            twist.linear.y = values[1]
            twist.linear.z = values[2]
            twist.angular.x = values[3]
            twist.angular.y = values[4]
            twist.angular.z = values[5]
            return twist

        self.fn = fn

        super(TwistNode, self).__init__(name=name, topic=topic,
                                        dimensions=self.dimensions,
                                        msg_type=Twist, trans_fnc=self.fn)


class MotionVWNode(RosPubNode):
    """
  This node publishes forward and angular velocity
  """

    def __init__(self, name, topic):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published
    """

        self.dimensions = 2

        def fn(values):
            twist = Twist()
            twist.linear.x = values[0]
            twist.angular.z = values[1]
            return twist

        self.fn = fn

        super(MotionVWNode, self).__init__(name=name, topic=topic,
                                           dimensions=self.dimensions,
                                           msg_type=Twist, trans_fnc=self.fn)


class MotionXYWNode(RosPubNode):
    """
  This node publishes translational and angular velocity
  """

    def __init__(self, name, topic):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published
    """

        self.dimensions = 3

        def fn(values):
            twist = Twist()
            twist.linear.x = values[index]
            twist.linear.y = values[index]
            twist.angular.z = values[index]
            return twist

        self.fn = fn

        super(MotionXYWNode, self).__init__(name=name, topic=topic,
                                            dimensions=self.dimensions,
                                            msg_type=Twist, trans_fnc=self.fn)


class OdometryNode(RosSubNode):
    """
  This node reads odometry data
  """

    # TODO: need to come up with a better way of specifying 'mask' as well as
    #      a better name for this parameter
    def __init__(self, name, topic, mask=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 use_quaternion=False):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    mask : list
        List of boolean representing which dimensions of the Odometry message
        are being subscribed to. All others will be ignored.
    """

        self.mask = mask
        self.dimensions = mask.count(True)

        def fn_quaternion(data):
            rval = [0] * self.dimensions

            index = 0

            if self.mask[0]:
                rval[index] = data.pose.pose.position.x
                index += 1
            if self.mask[1]:
                rval[index] = data.pose.pose.position.y
                index += 1
            if self.mask[2]:
                rval[index] = data.pose.pose.position.z
                index += 1

            if self.mask[3]:
                rval[index] = data.twist.twist.linear.x
                index += 1
            if self.mask[4]:
                rval[index] = data.twist.twist.linear.y
                index += 1
            if self.mask[5]:
                rval[index] = data.twist.twist.linear.z
                index += 1

            if self.mask[6]:
                rval[index] = data.pose.pose.orientation.x
                index += 1
            if self.mask[7]:
                rval[index] = data.pose.pose.orientation.y
                index += 1
            if self.mask[8]:
                rval[index] = data.pose.pose.orientation.z
                index += 1
            if self.mask[9]:
                rval[index] = data.pose.pose.orientation.w
                index += 1

            if self.mask[10]:
                rval[index] = data.twist.twist.angular.x
                index += 1
            if self.mask[11]:
                rval[index] = data.twist.twist.angular.y
                index += 1
            if self.mask[12]:
                rval[index] = data.twist.twist.angular.z
                index += 1

            return rval

        def fn_euler(data):
            rval = [0] * self.dimensions

            index = 0

            if self.mask[0]:
                rval[index] = data.pose.pose.position.x
                index += 1
            if self.mask[1]:
                rval[index] = data.pose.pose.position.y
                index += 1
            if self.mask[2]:
                rval[index] = data.pose.pose.position.z
                index += 1

            if self.mask[3]:
                rval[index] = data.twist.twist.linear.x
                index += 1
            if self.mask[4]:
                rval[index] = data.twist.twist.linear.y
                index += 1
            if self.mask[5]:
                rval[index] = data.twist.twist.linear.z
                index += 1

            x = data.pose.pose.orientation.x
            y = data.pose.pose.orientation.y
            z = data.pose.pose.orientation.z
            w = data.pose.pose.orientation.w

            quaternion = (x, y, z, w)
            euler = tf.transformations.euler_from_quaternion(quaternion)

            if self.mask[6]:
                rval[index] = euler[0]
                index += 1
            if self.mask[7]:
                rval[index] = euler[1]
                index += 1
            if self.mask[8]:
                rval[index] = euler[2]
                index += 1

            if self.mask[9]:
                rval[index] = data.twist.twist.angular.x
                index += 1
            if self.mask[10]:
                rval[index] = data.twist.twist.angular.y
                index += 1
            if self.mask[11]:
                rval[index] = data.twist.twist.angular.z
                index += 1

            return rval

        if use_quaternion:
            self.fn = fn_quaternion
        else:
            self.fn = fn_euler

        super(OdometryNode, self).__init__(name=name, topic=topic,
                                           dimensions=self.dimensions,
                                           msg_type=Odometry, trans_fnc=self.fn)


class PoseNode(RosSubNode):
    """
  This node reads pose data
  """

    # TODO: need to come up with a better way of specifying 'mask' as well as
    #      a better name for this parameter
    def __init__(self, name, topic, mask=[1, 1, 1, 1, 1, 1, ],
                 use_quaternion=False):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    mask : list
        List of boolean representing which dimensions of the Pose message
        are being subscribed to. All others will be ignored.
    """

        self.mask = mask
        self.dimensions = mask.count(True)

        def fn_quaternion(data):
            rval = [0] * self.dimensions

            index = 0

            if self.mask[0]:
                rval[index] = data.pose.position.x
                index += 1
            if self.mask[1]:
                rval[index] = data.pose.position.y
                index += 1
            if self.mask[2]:
                rval[index] = data.pose.position.z
                index += 1

            if self.mask[3]:
                rval[index] = data.pose.orientation.x
                index += 1
            if self.mask[4]:
                rval[index] = data.pose.orientation.y
                index += 1
            if self.mask[5]:
                rval[index] = data.pose.orientation.z
                index += 1
            if self.mask[6]:
                rval[index] = data.pose.orientation.w
                index += 1

            return rval

        def fn_euler(data):
            rval = [0] * self.dimensions

            index = 0

            if self.mask[0]:
                rval[index] = data.pose.position.x
                index += 1
            if self.mask[1]:
                rval[index] = data.pose.position.y
                index += 1
            if self.mask[2]:
                rval[index] = data.pose.position.z
                index += 1

            x = data.pose.orientation.x
            y = data.pose.orientation.y
            z = data.pose.orientation.z
            w = data.pose.orientation.w

            quaternion = (x, y, z, w)
            euler = tf.transformations.euler_from_quaternion(quaternion)

            if self.mask[3]:
                rval[index] = euler[0]
                index += 1
            if self.mask[4]:
                rval[index] = euler[1]
                index += 1
            if self.mask[5]:
                rval[index] = euler[2]
                index += 1

            return rval

        if use_quaternion:
            self.fn = fn_quaternion
        else:
            self.fn = fn_euler

        super(PoseNode, self).__init__(name=name, topic=topic,
                                       dimensions=self.dimensions,
                                       msg_type=PoseStamped, trans_fnc=self.fn)


class RotorcraftAttitudeNode(RosPubNode):
    """
  This node publishes roll, pitch, yaw, and thrust
  """

    def __init__(self, name, topic):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being published
    """

        self.dimensions = 4

        def fn(values):
            # ROS does not have a message type for rotorcraft attitude, so quaternion
            # is used because it has the same structure
            attitude = Quaternion()
            attitude.x = values[0]
            attitude.y = values[1]
            attitude.z = values[2]
            attitude.w = values[3]
            return attitude

        self.fn = fn

        super(RotorcraftAttitudeNode, self).__init__(name=name, topic=topic,
                                                     dimensions=self.dimensions,
                                                     msg_type=Quaternion,
                                                     trans_fnc=self.fn)


class SemanticCameraNode(RosSubNode):
    """
  This node is active when specific targets are seen by a semantic camera in
  MORSE. Each target is represented by one dimension of the output, and a 0.0
  means the target is not seen by the camera, and a 1.0 means that the target is
  with the field of view of the camera.
  """

    def __init__(self, name, topic, targets):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    targets : list
        List of str representing the names of the targets the camera is
        sensitive to
    """

        self.targets = targets
        self.dimensions = len(self.targets)

        def fn(data):
            rval = [0] * self.dimensions
            string = data.data
            # TODO: put in error handling for malformed string
            str_val = json.loads(string)
            if len(str_val) > 0:
                for i in str_val:
                    if i['name'] in self.targets:
                        rval[self.targets.index(i['name'])] = 1.0
                        break

            return rval

        self.fn = fn

        super(SemanticCameraNode, self).__init__(name=name, topic=topic,
                                                 dimensions=self.dimensions,
                                                 msg_type=String, trans_fnc=self.fn)


class VideoCameraNode(RosSubNode):
    """
  This node subsamples an image and outputs pixel intensity to the output
  dimensions
  """

    def __init__(self, name, topic):
        """
    Parameters
    ----------
    name : str
        An arbitrary name for the object
    topic : str
        The name of the ROS topic that is being subscribed to
    """

        from cv_bridge import CvBridge, CvBridgeError
        import numpy

        self.bridge = CvBridge()

        self.dimensions = 16 * 16

        def fn(data):
            rval = [0] * self.dimensions
            # cv_im = self.bridge.imgmsg_to_cv( data, "rgba8" )
            cv_im = self.bridge.imgmsg_to_cv(data, "mono8")
            # TODO: make sure mono conversion is correct
            im = numpy.array(cv_im)
            rval = list(im.ravel())
            return rval

        self.fn = fn

        super(VideoCameraNode, self).__init__(name=name, topic=topic,
                                              dimensions=self.dimensions,
                                              msg_type=Image, trans_fnc=self.fn)
