from network.network import Network
import tensorflow as tf
import numpy as np


class VGG16(Network):

    def __init__(self, input_shape, class_number, x, y, train=False, learning_rate=0.001):
        super().__init__()
        self.loss = None
        self.accuracy = None
        self._build_network(input_shape, class_number, train, learning_rate, x, y)

    def _build_network(self, network_input_shape, class_number, train, starter_learning_rate, x, y):
        self.x = x
        if train:
            self.keep_prob = 0.5
            self.y_ = y
            self.y = tf.one_hot(self.y_, class_number, 1.0, 0.0)
        self.conv1_1 = self.conv_layer('conv1_1', layer_input=self.x, shape=[3, 3, self.x.get_shape()[3].value,
                                                                                    64])
        self.conv1_2 = self.conv_layer('conv1_2', layer_input=self.conv1_1, shape=[3, 3, 64, 64])
        self.max_pool1 = self.max_pool(self.conv1_2)
        self.conv2_1 = self.conv_layer('conv2_1', layer_input=self.max_pool1, shape=[3, 3, 64, 128])
        self.conv2_2 = self.conv_layer('conv2_2', layer_input=self.conv2_1, shape=[3, 3, 128, 128])
        self.max_pool2 = self.max_pool(self.conv2_2)
        self.conv3_1 = self.conv_layer('conv3_1', layer_input=self.max_pool2, shape=[3, 3, 128, 256])
        self.conv3_2 = self.conv_layer('conv3_2', layer_input=self.conv3_1, shape=[3, 3, 256, 256])
        self.conv3_3 = self.conv_layer('conv3_3', layer_input=self.conv3_2, shape=[3, 3, 256, 256])
        self.max_pool3 = self.max_pool(self.conv3_3)
        self.conv4_1 = self.conv_layer('conv4_1', layer_input=self.max_pool3, shape=[3, 3, 256, 512])
        self.conv4_2 = self.conv_layer('conv4_2', layer_input=self.conv4_1, shape=[3, 3, 512, 512])
        self.conv4_3 = self.conv_layer('conv4_3', layer_input=self.conv4_2, shape=[3, 3, 512, 512])
        self.max_pool4 = self.max_pool(self.conv4_3)
        self.conv5_1 = self.conv_layer('conv5_1', layer_input=self.max_pool4, shape=[3, 3, 512, 512])
        self.conv5_2 = self.conv_layer('conv5_2', layer_input=self.conv5_1, shape=[3, 3, 512, 512])
        self.conv5_3 = self.conv_layer('conv5_3', layer_input=self.conv5_2, shape=[3, 3, 512, 512])
        self.max_pool5 = self.max_pool(self.conv5_3)
        self.flat_max_pool5 = tf.reshape(self.max_pool5, shape=[-1, 7*7*512])
        self.fc6 = self.fully_connected('fc6', self.flat_max_pool5, 4096)
        self.fc6 = tf.nn.relu(self.fc6)

        self.fc6 = tf.nn.dropout(self.fc6, keep_prob=self.keep_prob)
        self.fc7 = self.fully_connected('fc7', self.fc6, 4096)
        self.fc7 = tf.nn.relu(self.fc7)

        self.fc7 = tf.nn.dropout(self.fc7, keep_prob=self.keep_prob)
        self.fc8 = self.fully_connected('fc8', self.fc7, class_number)
        if train:
            self.global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                       decay_steps=100000, decay_rate=0.1, staircase=True)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.fc8))
            self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.fc8,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self, session):
        if self.loss is None:
            raise RuntimeError('Training a testing network!!')
        _, loss_value, accuracy_value = session.run([self.train_step, self.loss, self.accuracy])
        print('Loss {:.2f} Accuracy {:.2f}'.format(loss_value, accuracy_value))

    def test(self, session, batch, labels):
        if self.accuracy is None:
            raise RuntimeError('Cannot compute accuracy!!')
        accuracy = np.mean([session.run(self.accuracy, feed_dict={self.x: [batch[i]],
                                                                  self.y_: [labels[i]],
                                                                  self.keep_prob: 1.0})
                            for i in range(len(batch))])
        print('Accuracy: {:.2f}'.format(accuracy))

    def _restore_state(self, session):
        self.conv1_1 = self._restore_conv(session, 'conv1_1', layer_input=self.x)
        self.conv1_2 = self._restore_conv(session, 'conv1_2', layer_input=self.conv1_1)

        self.conv2_1 = self._restore_conv(session, 'conv2_1', layer_input=self.max_pool1)
        self.conv2_2 = self._restore_conv(session, 'conv2_2', layer_input=self.conv2_1)

        self.conv3_1 = self._restore_conv(session, 'conv3_1', layer_input=self.max_pool2)
        self.conv3_2 = self._restore_conv(session, 'conv3_2', layer_input=self.conv3_1)
        self.conv3_3 = self._restore_conv(session, 'conv3_3', layer_input=self.conv3_2)

        self.conv4_1 = self._restore_conv(session, 'conv4_1', layer_input=self.max_pool3)
        self.conv4_2 = self._restore_conv(session, 'conv4_2', layer_input=self.conv4_1)
        self.conv4_3 = self._restore_conv(session, 'conv4_3', layer_input=self.conv4_2)

        self.conv5_1 = self._restore_conv(session, 'conv5_1', layer_input=self.max_pool4)
        self.conv5_2 = self._restore_conv(session, 'conv5_2', layer_input=self.conv5_1)
        self.conv5_3 = self._restore_conv(session, 'conv5_3', layer_input=self.conv5_2)

        self.fc6 = self._restore_fully_connected(session, 'fc6', self.flat_max_pool5)
        self.fc6 = tf.nn.relu(self.fc6)
        self.fc6 = tf.nn.dropout(self.fc6, keep_prob=self.keep_prob)
        self.fc7 = self._restore_fully_connected(session,'fc7', self.fc6)
        self.fc7 = tf.nn.relu(self.fc7)
        self.fc7 = tf.nn.dropout(self.fc7, keep_prob=self.keep_prob)
        self.fc8 = self._restore_fully_connected(session,'fc8', self.fc7)



