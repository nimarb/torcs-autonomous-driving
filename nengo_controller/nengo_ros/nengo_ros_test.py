import nengo
import numpy as np
import nengo_ros
import rospy

model = nengo.Network(seed=2)
rospy.init_node('nengo_ros', anonymous=True, disable_signals=True)

with model:
  stim = nengo.Node(lambda x: np.sin(x))
  test_pub_node = nengo_ros.FloatPubNode(name='test', topic='/test/float')

  nengo.Connection(stim, test_pub_node, synapse=None)


if __name__ == '__main__':

  sim = nengo.Simulator(model)

  while True:
    sim.run(10)
