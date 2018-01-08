import nengo
import rospy
from driving_tasks import Steer, Accelerate, NonNeuralGear, Brake
from torcs_ros_network import TORCSInputNode, TORCSOutputNode

rospy.init_node('nengo_controller')

n_neurons = 600

model = nengo.Network()
with model:

    with nengo.Network():
        input_node = TORCSInputNode('TORCS_input', dnn=False)
        # input_ensemble = nengo.Ensemble(n_neurons=5000, dimensions=22)
        # angle, displ, speed
        input_steer_ensemble = nengo.Ensemble(n_neurons=4*n_neurons, dimensions=3)
        # 3 range
        input_ab_ensemble = nengo.Ensemble(n_neurons=4*n_neurons, dimensions=4)
        nengo.Connection(input_node[:3], input_steer_ensemble)
        nengo.Connection(input_node[2:6], input_ab_ensemble)

    with nengo.Network():
        output_node = TORCSOutputNode('CTRL_signals')

    # Steering
    Steer('steer', input_steer_ensemble, output_node, mode='report')

    # Accelerating
    Accelerate('accelerate', input_ab_ensemble, output_node, mode='report')

    # Brake
    Brake('brake', input_ab_ensemble, output_node, mode='report')

    # Gear
    NonNeuralGear('gear', input_node, output_node)

    # Clutch (Kupplung)
    # TODO: implement clutch

if __name__ == "__main__":
    print('done')
    sim = nengo.Simulator(model, progress_bar=False)
    sim.run(600)
