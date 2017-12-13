import nengo
import rospy
from driving_tasks import Steer, Accelerate, NonNeuralGear
from torcs_ros_network import TORCSInputNode, TORCSOutputNode

# TODO: everything has to be the same radius

# rospy.init_node('nengo_controller')

model = nengo.Network()
with model:

    with nengo.Network():
        input_node = TORCSInputNode('TORCS_input')
        input_ensemble = nengo.Ensemble(n_neurons=200, dimensions=22)
        nengo.Connection(input_node[:22], input_ensemble)

    with nengo.Network():
        output_node = TORCSOutputNode('CTRL_signals')

    # Steering
    Steer('steer', input_ensemble, output_node)

    # Accelerating
    Accelerate('accelerate', input_ensemble, output_node)

    # Brake
    # TODO: implement brake

    # Gear
    NonNeuralGear('gear', input_node, output_node)

    # Clutch (Kupplung)
    # TODO: implement clutch

if __name__ == "__main__":
    print('done')
    # sim = nengo.Simulator(model, progress_bar=True)
    # sim.run(120)
