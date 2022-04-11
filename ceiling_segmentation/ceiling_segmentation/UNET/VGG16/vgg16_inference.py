from soupsieve import select
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from UNET.VGG16.EncoderDecoder import EncoderDecoder
from PIL import Image
import numpy as np

### following lines are system dependent (mine 3080 in lab vs mine 960 laptop) 
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
config.gpu_options.polling_inactive_delay_msecs = 10
session = tf.compat.v1.Session(config=config)
#############################

class VGG16Inference:
    def __init__(self, num_class, image_size, weight_address) -> None:
        self.num_classes = num_class
        self.image_size = image_size
        self.weight_address = weight_address
        
    def build_model(self):
        self.encoderDecoder = EncoderDecoder(self.num_classes, batch_norm=False)
        self.encoderDecoder.build((None, self.image_size, self.image_size, 3))
        self.encoderDecoder.load_weights(self.weight_address)

    def create_mask(self, pred_mask: tf.Tensor) -> tf.Tensor:
        """Return a filter mask with the top 1 predictions
        only.

        Parameters
        ----------
        pred_mask : tf.Tensor
            A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
            N_CLASS values (vector) which represents the probability of the pixel
            being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
            has been predicted class 2 with a probability of 100%.

        Returns
        -------
        tf.Tensor
            A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
            for each pixels.
        """
        # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
        # 1 prediction for each class but we want the highest score only
        # so we use argmax
        pred_mask = tf.argmax(pred_mask, axis=-1)
        # pred_mask becomes [IMG_SIZE, IMG_SIZE]
        # but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
        pred_mask = tf.expand_dims(pred_mask, axis=-1)
        return pred_mask

    def display_sample(self, display_list):
        """Show side-by-side an input image,
        the ground truth and the prediction.
        """
        plt.figure(figsize=(18, 18))

        title = ['Input Image', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            img = tf.keras.preprocessing.image.array_to_img(display_list[i])
            plt.imshow(img)
            plt.axis('off')
        plt.show()


    def normalize(self, input_image: tf.Tensor) -> tf.Tensor:
        input_image = tf.cast(input_image, tf.float32)
        normalized_image = tf.image.per_image_standardization(input_image)
        return normalized_image

    def inference_from_file(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, (self.image_size, self.image_size))
        img = self.normalize(img)
        img = tf.keras.backend.expand_dims(img, 0)

        pred = self.encoderDecoder(img, training=False)
        pred_mask = self.create_mask(pred)

        # image = tf.keras.preprocessing.image.array_to_img(pred_mask[0])
        # image.save("predict.jpg")

        self.display_sample([img[0], pred_mask[0]])

    def inference(self, *cv_images):
        """
         receives a batch of cv2 images (batch_size = num of overhead cameras)
         and returns the segmented images in cv2 format
        :param cv_images: batch of images
        :return: batch of segmented images
        """

        tf_images = tf.convert_to_tensor(list(cv_images), dtype=tf.float32)
        normalized_images = self.normalize(tf_images)
        pred = self.encoderDecoder(normalized_images, training=False)
        pred_mask = tf.argmax(pred, axis=-1)
        return pred_mask.numpy()
