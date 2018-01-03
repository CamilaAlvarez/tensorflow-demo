import yaml
import os
import argparse
import tensorflow as tf
import numpy as np
from data.clothing_data_parser import ClothingTrainingData, ClothingValidationData
from network.factory import NetworkFactory

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config', help='YAML config file', required=True)
args = parser.parse_args()
classes = []
with open(args.config, 'r') as config_file:
    config = yaml.load(config_file)
    output_dir = config['base-output-dir']
    gpu_id = 0
    if 'gpu' in config:
        gpu_id = config['gpu']
    data = config['data-location']
    classes_file = config['classes-file']
    with open(classes_file) as classes_list:
        for line in classes_list:
            classes.append(line.strip())
    max_epochs = int(config['max-epochs'])
    base_model = None
    graph_model = None
    numpy_model = False
    if 'base-model' in config:
        base_model = config['base-model']
        if 'use-numpy' in config and config['use-numpy']:
            numpy_model = True
        elif 'graph-model' in config:
            graph_model = config['graph-model']

    network_name = config['network']
    data_queue = tf.train.string_input_producer([data], num_epochs=max_epochs)
    input_shape = (224, 224, 3)
    with tf.device('/gpu:{}'.format(gpu_id)):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(data_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'height': tf.FixedLenFeature([], tf.int64),
                                               'width': tf.FixedLenFeature([], tf.int64),
                                               'depth': tf.FixedLenFeature([], tf.int64)
                                           })
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        depth = tf.cast(features['depth'], tf.int32)
        image = tf.decode_raw(features['image_raw'], np.float32)
        image = tf.reshape(image, tf.stack(list(input_shape)))
        image = tf.cast(image, tf.float32)
        label = tf.cast(features['label'], tf.int32)
        batch_size = 64
        images_batch, labels_batch = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, capacity=50000, min_after_dequeue=1000)
        network = NetworkFactory.get_network(network_name)(input_shape, len(classes), images_batch, labels_batch,
                                                           train=True, learning_rate=0.001)
        init = tf.global_variables_initializer()
        session.run(init)
        init = tf.local_variables_initializer()
        session.run(init)
        if base_model is not None:
            if numpy_model:
                network.load_weights_np(base_model, session)
            elif graph_model is not None:
                network.load_state(session, graph_model, base_model)
            elif graph_model is None:
                print('Missing graph structure')
        iteration_step = 10000

        saver = tf.train.Saver()
        network.saver = saver
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=session, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                step += 1
                print('Step {}'.format(step))
                network.train(session)
                if step % iteration_step == 0:
                    network.save_state(session, os.path.join(output_dir, 'vgg16_model'), global_step=step)
        except tf.errors.OutOfRangeError:
            print('Done training for {} epochs, {} steps'.format(max_epochs, step))
        finally:
            coord.request_stop()
        coord.join(threads)
        network.save_state(session, os.path.join(output_dir, 'final_vgg16_model'))
        session.close()


