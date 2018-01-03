from PIL import Image
from abc import ABC, abstractmethod
import random
import numpy as np


class ClothingDataManager(ABC):

    def __init__(self, data_file, classes_file, batch_size, image_size, mean_pixel):
        self.batch_size = batch_size
        self.mean_pixel = mean_pixel
        self.image_size = image_size
        self.epochs_completed = 0
        self._load_classes(classes_file)
        self._load_data_file(data_file)

    def _load_classes(self, classes_file):
        self.classes = []
        with open(classes_file) as file:
            for line in file:
                klass = line.strip()
                self.classes.append(klass)
        self.number_classes = len(self.classes)

    def _load_data_file(self, data_file):
        self.images = []
        self.current_batch_index = 0
        with open(data_file) as data:
            for line in data:
                line = line.strip()
                line_data = line.split('\t')
                self.images.append(ClothingImage(line_data[0], line_data[1], self.classes.index(line_data[2])))
        random.shuffle(self.images)

    def _prepare_image(self, image):
        resized_image = np.array(image.resize(self.image_size[0:-1], Image.BICUBIC), dtype=np.float32)
        resized_image -= self.mean_pixel
        return resized_image

    @abstractmethod
    def next_batch(self, max_epochs=1):
        raise NotImplementedError


class ClothingTrainingData(ClothingDataManager):

    def next_batch(self, max_epochs=1):
        number_images = len(self.images)
        used_images = 0
        while self.epochs_completed < max_epochs:
            # Load and resize the images
            final_index = min(self.current_batch_index + self.batch_size, number_images)
            batch_images = self.images[self.current_batch_index: final_index]
            if final_index < (self.current_batch_index + self.batch_size):
                for i in range(0, self.current_batch_index + self.batch_size - final_index):
                    batch_images.append(self.images[i])
            number_batch_images = len(batch_images)
            labels = np.zeros((number_batch_images, self.number_classes), dtype=np.float32)
            loaded_batch_images = np.zeros((number_batch_images,) + self.image_size)
            for i,batch_image in enumerate(batch_images):
                image = Image.open(batch_image.path)
                resized_image = self._prepare_image(image)
                labels[i, batch_image.klass] = 1.
                loaded_batch_images[i,:] = resized_image
            self.current_batch_index = (self.current_batch_index + self.batch_size) % number_images
            used_images += number_batch_images
            if (used_images / number_images) > self.epochs_completed:
                self.epochs_completed = int(used_images / number_images)
            yield loaded_batch_images, labels


class ClothingValidationData(ClothingDataManager):

    def next_batch(self, max_epochs=1):
        images_range = range(len(self.images))
        while True:
            picked_indexes = random.sample(images_range, self.batch_size)
            labels = np.zeros((self.batch_size, self.number_classes), dtype=np.float32)
            loaded_batch_images = np.zeros((self.batch_size,) + self.image_size)
            for i, index in enumerate(picked_indexes):
                batch_image = self.images[index]
                try:
                    image = Image.open(batch_image.path)
                    resized_image = self._prepare_image(image)
                    labels[i, batch_image.klass] = 1.
                    loaded_batch_images[i, :] = resized_image
                except IOError:
                    print('Could not open image {}'.format(batch_image.path))
            yield loaded_batch_images, labels


class ClothingImage:

    def __init__(self, image_id, path, klass):
        self.image_id = image_id
        self.path = path
        self.klass = klass