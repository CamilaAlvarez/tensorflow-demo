import os
import random
import argparse
import tensorflow as tf
from skimage import transform, io
import numpy as np
import yaml
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--config', dest='config_file', required=True, help='YAML config file')
parser.add_argument('--log', dest='log_level', default=logging.WARNING, help='Logging level')
args = parser.parse_args()
logging.basicConfig(format='%(levelname)s:%(message)s', level=args.log_level)
classes = []
images = []
with open(args.config_file, 'r') as config_file:
    config = yaml.load(config_file)
    output_file = config['output-file']
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    classes_file = config['class-list']
    images_file = config['image-list']
    red_mean_pixel = float(config['mean-pixel']['r'])
    green_mean_pixel = float(config['mean-pixel']['g'])
    blue_mean_pixel = float(config['mean-pixel']['b'])
    new_image_size = (int(config['image-size']['width']), int(config['image-size']['height']))
    with open(classes_file) as classes_list, open(images_file) as image_list:
        for class_line in classes_list:
            classes.append(class_line.strip())
        for image_line in image_list:
            image_info = image_line.strip().split('\t')
            klass = image_info[2]
            if klass not in classes:
                logging.warning('{} not in class list'.format(klass))
                logging.warning('Skipping image: {}'.format(image_info[1]))
                continue
            images.append({'path':image_info[1], 'label': classes.index(image_info[2])})
    random.shuffle(images)
    writer = tf.python_io.TFRecordWriter(output_file)
    logging.info('Opening writer {}'.format(output_file))
    for image_info in images:
        try:
            image = io.imread(image_info['path'])
            resized_image = transform.resize(image, new_image_size)
            resized_image = resized_image - [red_mean_pixel, green_mean_pixel, blue_mean_pixel]
            resized_image = resized_image.astype(np.float32)
            height, width, channels = resized_image.shape
            #channels = resized_image.split()
            #substract_mean_pixel = lambda mean_value: lambda pixel_value: pixel_value - mean_value
            #if len(channels) < 3:
            #    logging.warning('Non color image found: {}'.format(image_info['path']))
            #    final_image = Image.eval(channels[0], substract_mean_pixel(red_mean_pixel))
            #else:
            #    final_image_red = Image.eval(channels[0], substract_mean_pixel(red_mean_pixel))
            #    final_image_green = Image.eval(channels[1], substract_mean_pixel(green_mean_pixel))
            #    final_image_blue = Image.eval(channels[2], substract_mean_pixel(blue_mean_pixel))
            #    final_image = Image.merge('RGB', [final_image_red, final_image_green, final_image_blue])
            #image_data = io.BytesIO()
            #final_image.save(image_data, format=image.format)
            #image_bytes = image_data.getvalue()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[image_info['label']])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[resized_image.tostring()]))
            }))
            logging.info('Writing image to record: {}'.format(image_info['path']))
            writer.write(example.SerializeToString())
        except IOError:
            logging.warning('Image not found: {}'.format(image_info['path']))
    logging.info('Closing writer')
    writer.close()




