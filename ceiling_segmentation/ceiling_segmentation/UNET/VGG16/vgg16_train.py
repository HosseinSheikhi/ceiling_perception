import tensorflow as tf
from ceiling_segmentation.UNET.VGG16.EncoderDecoder import EncoderDecoder
from ceiling_segmentation.utils.LoadData import LoadData
import matplotlib.pyplot as plt
import datetime
import numpy as np
import pathlib
import os

### following lines are system dependent (mine 3080 in lab vs mine 960 laptop) 
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
config.gpu_options.polling_inactive_delay_msecs = 10
session = tf.compat.v1.Session(config=config)
#############################

class VGG16Train:
    def __init__(self, image_size, num_channels, num_classes, batch_size, buffer_size, epoch, seed, data_address):
        self.batch_size = batch_size
        self.image_size = image_size
        self.buffer_size = buffer_size
        self.epoch = epoch
        self.autotune = tf.data.experimental.AUTOTUNE
        self.seed = seed
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.data_address = data_address

        # set up the metric and logs
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='train_accuracy')
        self.test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
        self.test_log_dir = 'logs/gradient_tape/' + self.current_time + '/test'
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)

        self.loss_function = None
        self.optimizer = None

    def load_data(self):
        self.dataset = LoadData(self.data_address + "/training/images/*.png",
                           self.data_address + "/validation/images/*.png",
                           self.image_size, self.batch_size, shuffle_buffer_size=self.buffer_size,
                           seed=self.seed).get_dataset()

        # following lines are used for debug
        print(self.dataset['train'])
        print(self.dataset['val'])

        sample_image = None
        sample_mask = None
        for image, segmented_mask in self.dataset['train'].take(1):
            sample_image, sample_mask = image, segmented_mask

        self.display_sample([sample_image[0], sample_mask[0]])

    def display_sample(self, display_list):
        """
        Show side-by-side an input image, the ground truth and the prediction.
        :param display_list: a list including [image, ground truth] or [image, ground truth, prediction]
        :return:
        """
        plt.figure(figsize=(18, 18))

        title = ['Input Image', 'True Mask', 'Predicted Mask']

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i + 1)
            plt.title(title[i])
            img = tf.keras.preprocessing.image.array_to_img(display_list[i])
            plt.imshow(img)
            plt.axis('off')
        plt.show()

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

    def show_predictions(self, dataset, num=1):
        """Show a sample prediction.

        Parameters
        ----------
        dataset : [type], optional
            [Input dataset, by default None
        num : int, optional
            Number of sample to show, by default 1
        """

        for image, segmented_mask in dataset.take(num):
            sample_image, sample_mask = image, segmented_mask

            # The UNET is expecting a tensor of the size
            # [BATCH_SIZE, IMG_SIZE, IMG_SIZE, 3]
            # but sample_image[0] is [IMG_SIZE, IMG_SIZE, 3]
            # and we want only 1 inference to be faster
            # so we add an additional dimension [1, IMG_SIZE, IMG_SIZE, 3]
            one_img_batch = sample_image[0][tf.newaxis, ...]
            pred_mask = self.encoderDecoder(one_img_batch, training=False)
            mask = self.create_mask(pred_mask)
            self.display_sample([sample_image[0], sample_mask[0], mask[0]])

    def weighted_loss_function(self, y_true, y_pred):
        cross_entropy = tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)
        # calculate weight
        y_true = tf.cast(y_true, dtype='float32')
        y_true = tf.where(y_true == 0, np.dtype('float32').type(0.25), y_true)
        weight = tf.where(y_true == 1, np.dtype('float32').type(0.75), y_true)
        # multiply weight by cross entropy
        weight = tf.squeeze(weight)
        weighted_cross_entropy = tf.multiply(weight, cross_entropy)
        return tf.reduce_mean(weighted_cross_entropy)

    def build_model(self):
        self.encoderDecoder = EncoderDecoder(self.num_classes, batch_norm=False)

        # freeze the encoder and initialize it weights by vgg trained on imagenet
        self.encoderDecoder.encoder.trainable = False
        self.encoderDecoder.build((None, self.image_size, self.image_size, 3))
        self.encoderDecoder.encoder.set_weights(tf.keras.applications.VGG16(include_top=False, weights='imagenet',
                                                                       input_shape=(
                                                                       self.image_size, self.image_size, 3)).get_weights())

        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-6)

        self.show_predictions(self.dataset['val'], 1)

    @tf.function
    def train_model(self, images, masks):
        with tf.GradientTape() as g:
            prediction = self.encoderDecoder(images)
            loss = self.loss_function(masks, prediction)

        trainable_variables = self.encoderDecoder.trainable_variables
        gradients = g.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.train_loss.update_state(loss)
        self.train_acc.update_state(masks, prediction)

    @tf.function
    def test_model(self, images, masks):
        predictions = self.encoderDecoder(images)
        loss = self.loss_function(masks, predictions)

        self.test_loss.update_state(loss)
        self.test_acc.update_state(masks, predictions)

    def train_procedure(self):
        batch_train_ctr = 0
        batch_test_ctr = 0

        for repeat in range(self.epoch):
            # reset the matrices at the beginning of every epoch
            self.train_loss.reset_states()
            self.train_acc.reset_states()
            self.test_loss.reset_states()
            self.test_acc.reset_states()

            for (x_batch, y_batch) in self.dataset['train']:
                self.train_model(x_batch, y_batch)
                batch_train_ctr += 1

                template = 'Epoch {}, Batch {}, Loss: {}, Accuracy: {}'
                print(template.format(repeat, batch_train_ctr,
                                    self.train_loss.result(),
                                    self.train_acc.result() * 100))

                with self.train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', self.train_loss.result(), step=batch_train_ctr)
                    tf.summary.scalar('train_accuracy', self.train_acc.result(), step=batch_train_ctr)

            for (x_batch, y_batch) in self.dataset['val']:
                self.test_model(x_batch, y_batch)
                batch_test_ctr += 1

                template = 'Epoch {}, Batch{}, Test Loss: {}, Test Accuracy: {}'
                print(template.format(repeat, batch_test_ctr,
                                    self.test_loss.result(),
                                    self.test_acc.result() * 100))

                with self.test_summary_writer.as_default():
                    tf.summary.scalar('test_loss', self.test_loss.result(), step=batch_test_ctr)
                    tf.summary.scalar('test_accuracy', self.test_acc.result(), step=batch_test_ctr)

            self.show_predictions(self.dataset['val'], num=5)
            self.encoderDecoder.save_weights(os.getcwd()+"/weights/WithoutBN/NaiveLoss"+str(repeat+1)+"/")








