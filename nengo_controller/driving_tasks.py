import nengo
import numpy as np
import os
import h5py


# TODO: NonNeuralGear Node - Ensemble - Ensemble - Node statt Node - Node?

class Steer(nengo.Network):
    def __init__(self, name, input_ensemble, output_node, mode='default'):
        super(Steer, self).__init__(label=name)

        with self:
            steer_ensemble = nengo.Ensemble(n_neurons=200, dimensions=1)

        nengo.Connection(steer_ensemble, output_node[4])

        if mode == 'default':
            nengo.Connection(input_ensemble, steer_ensemble, function=lambda x: x[0] - 0.5 * x[1])
        else:
            raise NameError('wrong mode')


class Accelerate(nengo.Network):
    def __init__(self, name, input_ensemble, output_node, mode='default'):
        super(Accelerate, self).__init__(label=name)

        with self:
            accel_ensemble = nengo.Ensemble(n_neurons=200, dimensions=1)

        nengo.Connection(accel_ensemble, output_node[0])

        if mode == 'default':
            # TODO: implement default accelerate function
            nengo.Connection(input_ensemble, accel_ensemble, function=lambda x: 0)
        else:
            raise NameError('wrong mode')


class NonNeuralGear(nengo.Network):
    def __init__(self, name, input_node, output_node, gear_up=[5000, 6000, 6000, 6500, 7000, 0],
                 gear_down=[0, 2500, 3000, 3000, 3500, 3500]):
        super(NonNeuralGear, self).__init__(label=name)

        with self:
            self.gear_up = gear_up
            self.gear_down = gear_down

            def get_gear(data):
                rpm = data[23]
                gear = int(np.round(data[22]))

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

            nengo.Connection(input_node, output_node[3], function=get_gear)

