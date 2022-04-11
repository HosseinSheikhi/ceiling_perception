import tensorflow as tf
import numpy as np


class LoadData:
    def __init__(self, training_path, validation_path, image_size, batch_size, shuffle_buffer_size, seed):
        """
        put the dataset in the following format:
        - training
          -- images
             --- image_00001_img.png
          -- masks
             --- image_00001_layer.png
        - validation
          -- images
            --- image_10001_img.png
          -- masks
            --- image_10001_layer.png
        :param training_path: path to the training images to the format ../training/images/*.png
        :param validation_path: path to the validation images to the format ../validation/images/*.png
        :param image_size: desire image size for feeding to the network
        :param batch_size: batch size for training
        :param shuffle_buffer_size: buffer size for shuffling dataset. Better to be as same as training dataset size
        :param seed: random seed for shuffling
        """
        self.training_path = training_path
        self.validation_path = validation_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.buffer_size = shuffle_buffer_size
        self.seed = seed

    def get_dataset(self):
        train_files = tf.data.Dataset.list_files(self.training_path)
        train_files = train_files.shuffle(buffer_size=self.buffer_size, seed=self.seed)
        train_dataset = train_files.map(self.parse_image)

        val_files = tf.data.Dataset.list_files(self.validation_path)
        val_files = val_files.shuffle(buffer_size=self.buffer_size, seed=self.seed)
        val_dataset = val_files.map(self.parse_image)
        dataset = {"train": train_dataset, "val": val_dataset}

        # -- Train Dataset --#
        dataset['train'] = dataset['train'].map(self.load_image_train, num_parallel_calls=2)
        dataset['train'] = dataset['train'].batch(self.batch_size)
        dataset['train'] = dataset['train'].prefetch(buffer_size=1)  # will prefetch 1 batch

        # -- Validation Dataset --#
        dataset['val'] = dataset['val'].map(self.load_image_test, num_parallel_calls=2)
        dataset['val'] = dataset['val'].batch(self.batch_size*4)
        dataset['val'] = dataset['val'].prefetch(buffer_size=1)  # will prefetch 1 batch size
        return dataset

    def parse_image(self, image_path: str) -> dict:
        # load the raw data from the file as a string
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.uint8)

        # acquiring corresponding mask_path by changing the address
        mask_path = tf.strings.regex_replace(image_path, "images", "masks")
        mask_path = tf.strings.regex_replace(mask_path, "img", "layer")
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.uint8)

        # mask contains a class index for each pixel
        # we have two class : background:0, obstacle:8
        # for the sake of simplicity and consistency we would like to have: background:0  Obstacle:1
        # The original images includes some pixels equal to 4 for boundary of objects. we treat them as background
        mask = tf.where(mask == 4, np.dtype('uint8').type(0), mask)
        mask = tf.where(mask == 8, np.dtype('uint8').type(1), mask)
        return {'image': img, 'mask': mask}

    @tf.function
    def normalize(self, input_image: tf.Tensor) -> tf.Tensor:
        input_image = tf.cast(input_image, tf.float32)
        normalized_image = tf.image.per_image_standardization(input_image)
        return normalized_image

    @tf.function
    def load_image_train(self, train_data: dict) -> tuple:
        input_image = tf.image.resize(train_data['image'], (self.image_size, self.image_size),
                                      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_mask = tf.image.resize(train_data['mask'], (self.image_size, self.image_size),
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        #some data augmentation can be done here
        # if tf.random.uniform(()) > 0.25:
        #     input_image = tf.image.random_hue(input_image, 0.08)
        #     input_image = tf.image.random_saturation(input_image, 0.6, 1.6)
        #     input_image = tf.image.random_brightness(input_image, 0.05)
        #     input_image = tf.image.random_contrast(input_image, 0.7, 1.3)

        input_image = self.normalize(input_image)
        return input_image, input_mask

    @tf.function
    def load_image_test(self, test_data: dict) -> tuple:
        input_image = tf.image.resize(test_data['image'], (self.image_size, self.image_size),
                                      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        input_mask = tf.image.resize(test_data['mask'], (self.image_size, self.image_size),
                                     tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        input_image = self.normalize(input_image)
        return input_image, input_mask