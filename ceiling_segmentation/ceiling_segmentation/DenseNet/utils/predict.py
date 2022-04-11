import tensorflow as tf
import os
import matplotlib.pyplot as plt
from DenseNet.models.FC_DenseNet import FCDenseNet
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

NUM_CLASS = 2
IMAGE_SIZE = 224
model = FCDenseNet(NUM_CLASS, 103, "Test")
model.load_weights(
    "/home/hossein/FloorSegmentation/ImageSegmentation/DenseNet/utils/weights/103layers_6/")


def display_sample(display_list):
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


def normalize(input_image: tf.Tensor) -> tf.Tensor:
    input_image = tf.cast(input_image, tf.float32)
    normalized_image = tf.image.per_image_standardization(input_image)
    return normalized_image


img = tf.io.read_file("/home/hossein/FloorSegmentation/ImageSegmentation/DenseNet/testImages/first_overhead.jpg")
img = tf.image.decode_png(img, channels=3)
img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
img = normalize(img)
img = tf.keras.backend.expand_dims(img, 0)

pred = model(img, training=False)
pred_mask = tf.argmax(pred, axis=-1)
# pred_mask becomes [IMG_SIZE, IMG_SIZE]
# but matplotlib needs [IMG_SIZE, IMG_SIZE, 1]
pred_mask = tf.expand_dims(pred_mask, axis=-1)
display_sample([img[0], pred_mask[0]])
