import nengo
import numpy as np
import os
import h5py


class Steer(nengo.Network):
    def __init__(self, torcs_network, b_direct=False, b_probe=False):
        super(Steer, self).__init__()
        if b_direct:
            self.config[nengo.Ensemble].neuron_type = nengo.Direct()

        with self:
            # representational neurons
            self.angle_neurons = nengo.Ensemble(n_neurons=200, dimensions=1)
            self.trackPos_neurons = nengo.Ensemble(n_neurons=200, dimensions=1)
            self.scan_track = nengo.Ensemble(n_neurons=100*19, dimensions=19, radius=2)

            # output neurons
            self.steer_neurons = nengo.Ensemble(n_neurons=200, dimensions=1)

            if b_probe:
                self.p_steer = nengo.Probe(self.steer_neurons, synapse=0.01)

            nengo.Connection(self.angle_neurons, self.steer_neurons, transform=1)
            nengo.Connection(self.trackPos_neurons, self.steer_neurons, transform=-0.5)

        nengo.Connection(torcs_network.torcs_sensors_state[0], self.angle_neurons)
        nengo.Connection(torcs_network.torcs_sensors_state[3], self.trackPos_neurons)
        nengo.Connection(torcs_network.scan_track, self.scan_track, transform=1/100.0)
        nengo.Connection(self.steer_neurons, torcs_network.torcs_ctrl_cmd[4])

class SteerRanges(nengo.Network):
    def __init__(self, torcs_network, data_file = '/home/flo/data/torcs/train_steer_aplina1.h5', b_direct=False, b_probe=False):
        super(SteerRanges, self).__init__()
        if b_direct:
            self.config[nengo.Ensemble].neuron_type = nengo.Direct()

        with self:
            self.data = {}
            if os.path.isfile(data_file):
                with h5py.File(data_file,'r') as hf:
                    for key in hf.keys():
                        self.data[key] = np.array(hf.get(key))
            # neuron ensembles to calculate steering value
            self.scan_track = nengo.Ensemble(n_neurons=100*19, dimensions=19, radius=2)

            self.steering = nengo.Ensemble(n_neurons=100, dimensions=1)
            if b_probe:
                self.p_steer = nengo.Probe(self.steering, synapse=0.01)
            desired_steering = self.data['angle'] - 0.5*self.data['track_pos']
            desired_steering.shape = desired_steering.shape[0],1
            self.conn = nengo.Connection(self.scan_track, self.steering, function=desired_steering, eval_points=self.data['scan_track']/100.0)

        nengo.Connection(torcs_network.scan_track, self.scan_track, transform=1/100.0)
        nengo.Connection(self.steering, torcs_network.torcs_ctrl_cmd[4])

class Accelerate(nengo.Network):
    def __init__(self, torcs_network, b_direct=False):
        super(Accelerate, self).__init__()
        if b_direct:
            self.config[nengo.Ensemble].neuron_type = nengo.Direct()

        with self:

            # neuron ensembles to calculate acceleration values
            self.speed_neurons = nengo.Ensemble(n_neurons=400, dimensions=2, radius=1.5)
            self.accel_neurons = nengo.Ensemble(n_neurons=200, dimensions=1, radius=1.5)

            def calc_accel_func(x):
                accel = x[1]
                if x[0] < 0.6:
                    accel += 0.1
                    if accel > 1:
                        accel = 1.0
                else:
                    accel -= 0.1
                    if accel < 0:
                        accel = 0.0

                return accel

        nengo.Connection(self.speed_neurons, self.accel_neurons, function=calc_accel_func)

        nengo.Connection(torcs_network.speed[0], self.speed_neurons[0], transform=1 / 100.0)
        nengo.Connection(torcs_network.torcs_ctrl_state[0], self.speed_neurons[1])
        nengo.Connection(self.accel_neurons, torcs_network.torcs_ctrl_cmd[0])


class NonNeuralGear(nengo.Network):
    def __init__(self, torcs_network, gear_up=[5000, 6000, 6000, 6500, 7000, 0],
                 gear_down=[0, 2500, 3000, 3000, 3500, 3500], b_direct=False):
        super(NonNeuralGear, self).__init__()
        if b_direct:
            self.config[nengo.Ensemble].neuron_type = nengo.Direct()

        with self:
            self.gear_up = gear_up
            self.gear_down = gear_down

            self.input = nengo.Ensemble(n_neurons=200, dimensions=2, neuron_type=nengo.Direct())

            def get_gear(data):
                rpm = data[0]
                gear = int(np.round(data[1]))

                desired_gear = gear

                if gear <= 0:
                    desired_gear = 1
                else:
                    if gear < 6 and rpm >= self.gear_up[gear - 1]:
                        desired_gear = gear + 1
                    else:
                        if gear > 1 and rpm <= self.gear_down[int(gear) - 1]:
                            desired_gear = gear - 1

                return desired_gear

            self.desired_gear = nengo.Ensemble(n_neurons=100, dimensions=1, neuron_type=nengo.Direct())

            nengo.Connection(self.input, self.desired_gear, function=get_gear)

        # get current rpm value
        nengo.Connection(torcs_network.torcs_sensors_state[2], self.input[0])
        # get gear value
        nengo.Connection(torcs_network.torcs_sensors_state[1], self.input[1])
        # set gear value back to torcs_network
        nengo.Connection(self.desired_gear, torcs_network.torcs_ctrl_cmd[3])
