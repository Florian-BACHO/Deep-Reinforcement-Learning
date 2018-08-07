import tensorflow as tf
import numpy as np
import os

def getLayerVariables(layer):
    w = tf.get_default_graph().get_tensor_by_name(os.path.split(layer.name)[0] + '/kernel:0')
    b = tf.get_default_graph().get_tensor_by_name(os.path.split(layer.name)[0] + '/bias:0')
    return w, b

# Deep Convolutional Q-Network
class DCQN:
    def __init__(self, input_shape, nb_out):
        self.nb_out = nb_out
        self.entry = tf.placeholder(tf.float32, tuple([None] + list(input_shape)))
        self.layers = []

        conv = self._makeConvLayers(self.entry)

        # Reshape convolution to (None, <conv_size>)
        flattenConv = tf.contrib.layers.flatten(conv)

        self.out = self._makeFullyConnectedLayers(flattenConv, nb_out)

    def buildLearningTensors(self, learning_rate):
        self.expectedQValues = tf.placeholder(tf.float32, (None))

        self.actions_indices = tf.placeholder(tf.int32, (None, 2))
        actions_tensors = tf.gather_nd(self.out, self.actions_indices)

        loss = tf.losses.mean_squared_error(self.expectedQValues, actions_tensors)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.training_op = optimizer.minimize(loss)

    # Copy model weights and bias to other DCQN
    def copyWeights(self, session, dest):
        for i in range(len(self.layers)):
            w, b = getLayerVariables(self.layers[i])
            w_dest, b_dest = getLayerVariables(dest.layers[i])
            session.run(tf.assign(w_dest, w))
            session.run(tf.assign(b_dest, b))

    def _makeConvLayers(self, entry):
        convLayer1 = tf.layers.conv2d(entry, 32, 8, [4, 4], padding="SAME", activation=tf.nn.relu)
        self.layers.append(convLayer1)
        convLayer2 = tf.layers.conv2d(convLayer1, 64, 4, [2, 2], padding="SAME", activation=tf.nn.relu)
        self.layers.append(convLayer2)
        convLayer3 = tf.layers.conv2d(convLayer2, 64, 3, [1, 1], padding="SAME", activation=tf.nn.relu)
        self.layers.append(convLayer3)
        return convLayer3

    def _makeFullyConnectedLayers(self, entry, nbOut):
        fc1 = tf.layers.dense(entry, 512, tf.nn.relu)
        self.layers.append(fc1)
        fc2 = tf.layers.dense(fc1, nbOut)
        self.layers.append(fc2)
        return fc2

    def activate(self, session, x):
        return session.run(self.out, feed_dict={self.entry: x})

    def train(self, session, batch, targetNet, discountRate):
        states, actions, rewards, dones, next_states = batch

        next_qvalues = targetNet.activate(session, next_states).max(axis=1)
        next_qvalues[np.nonzero(dones)] = 0.

        expected_qvalues = next_qvalues * discountRate + rewards

        actions_indices = np.array([[i, idx] for i, idx in enumerate(actions)])

        session.run(self.training_op, feed_dict={self.entry: states, \
                                                 self.actions_indices: actions_indices, \
                                                 self.expectedQValues: expected_qvalues})
