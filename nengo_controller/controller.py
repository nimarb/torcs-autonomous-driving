import nengo
import rospy
from driving_tasks import Steer, SteerRanges, Accelerate, NonNeuralGear
from torcs_ros_network import TORCSROSNetwork

rospy.init_node('nengo_controller')

model = nengo.Network()
with model:
    Nengo_ROS = TORCSROSNetwork()

    # Steering
    task_desired_steer = Steer(Nengo_ROS)

    # Accelerating
    task_accel = Accelerate(Nengo_ROS)

    # Gear
    task_gear = NonNeuralGear(Nengo_ROS)

    # Clutch (Kupplung)

if __name__ == "__main__":
    sim = nengo.Simulator(model, progress_bar=True)
    sim.run(120)
