from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class Network(ABC):

    def __init__(self):
        self.parameters = []
        self._saver = None

    @property
    def saver(self):
        return self._saver

    @saver.setter
    def saver(self, saver):
        self._saver = saver

    @staticmethod
    def _weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _bias_variable(shape, name):
        initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
        return tf.Variable(initial, name=name)

    @staticmethod
    def _conv2d(x, W, stride, padding):
        return tf.nn.conv2d(x, W, strides=stride, padding=padding)

    def fully_connected(self, name, layer_input, size):
        input_size = int(layer_input.get_shape()[1])
        W = self._weight_variable([input_size, size], name)
        b = self._bias_variable([size], '{}_bias'.format(name))
        self.parameters += [W, b]
        return tf.matmul(layer_input, W) + b

    def conv_layer(self, name, layer_input, shape, stride=list([1, 1, 1, 1]), padding='SAME'):
        W = self._weight_variable(shape, name)
        b = self._bias_variable([shape[3]],'{}_bias'.format(name))
        self.parameters += [W, b]
        return tf.nn.relu(self._conv2d(layer_input, W, stride, padding) + b)

    def _restore_conv(self, session, name, layer_input, stride=list([1, 1, 1, 1]), padding='SAME'):
        W = session.run('{}:0'.format(name))
        b = session.run('{}_bias:0'.format(name))
        return tf.nn.relu(self._conv2d(layer_input, W, stride, padding) + b)

    def _restore_fully_connected(self, session, name, layer_input):
        W = session.run('{}:0'.format(name))
        b = session.run('{}_bias:0'.format(name))
        return tf.matmul(layer_input, W) + b

    @staticmethod
    def max_pool(layer_input, size=list([1,2,2,1]), strides=list([1,2,2,1])):
        # [batch, height, width, channels]
        return tf.nn.max_pool(layer_input, strides=strides, ksize=size, padding='SAME')

    @abstractmethod
    def train(self, session):
        raise NotImplementedError

    @abstractmethod
    def test(self, session, batch, labels):
        raise NotImplementedError

    def load_weights_np(self, weights_file, session):
        if weights_file is None:
            return
        weights = np.load(weights_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print(i, k, np.shape(weights[k]))
            if np.shape(self.parameters[i]) != np.shape(weights[k]):
                print('Skipping weights for layer with shape:{}'.format(np.shape(self.parameters[i])))
                continue
            session.run(self.parameters[i].assign(weights[k]))

    @abstractmethod
    def _restore_state(self, session):
        raise NotImplementedError

    """
    Restores a graph described in a graph file and a model file so its variables can be accessed
    """
    def load_state(self, session, graph_file, model_file):
        tf.reset_default_graph()
        imported_graph = tf.train.import_meta_graph(graph_file)
        imported_graph.restore(session, model_file)
        self._restore_state(session)

    def save_state(self, session, path, global_step=None):
        self.saver.save(session, path, global_step=global_step)