import tensorflow as tf
from DenseNet.models.FC_DenseNet import FCDenseNet
import matplotlib.pyplot as plt
import LoadData
import datetime
import os
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

BATCH_SIZE = 4
IMAGE_SIZE = 224
BUFFER_SIZE = 10
AUTOTUNE = tf.data.experimental.AUTOTUNE
SEED = 25
N_CHANNELS = 3
N_CLASSES = 2
EPOCHS = 6

dataset = LoadData.LoadData("/home/hossein/FloorSegmentation/synthesisData/training/images/*.png",
                            "/home/hossein/FloorSegmentation/synthesisData/validation/images/*.png",
                            IMAGE_SIZE, BATCH_SIZE, shuffle_buffer_size=BUFFER_SIZE, seed=123).get_dataset()
print(dataset['train'])
print(dataset['val'])


def display_sample(display_list):
    """Show side-by-side an input image,
    the ground truth and the prediction.
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


for image, segmented_mask in dataset['train'].take(1):
    sample_image, sample_mask = image, segmented_mask

display_sample([sample_image[0], sample_mask[0]])


def create_mask(pred_mask: tf.Tensor) -> tf.Tensor:
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


def show_predictions(dataset, num=1):
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
        pred_mask = encoderDecoder(one_img_batch, training=False)
        mask = create_mask(pred_mask)
        display_sample([sample_image[0], sample_mask[0], mask[0]])


encoderDecoder = FCDenseNet(N_CLASSES, 103, 'Train')


loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, epsilon=1e-6)

# set up the metric and logs
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='train_accuracy')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

show_predictions(dataset['val'], 1)


@tf.function
def train_model(images, masks):
    with tf.GradientTape() as g:
        prediction = encoderDecoder(images)
        loss = loss_function(masks, prediction)

    trainable_variables = encoderDecoder.trainable_variables
    gradients = g.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    train_loss.update_state(loss)
    train_acc.update_state(masks, prediction)


@tf.function
def test_model(images, masks):
    predictions = encoderDecoder(images)
    loss = loss_function(masks, predictions)

    test_loss.update_state(loss)
    test_acc.update_state(masks, predictions)


batch_train_ctr = 0
batch_test_ctr = 0
for repeat in range(EPOCHS):

    # reset the matrices at the beginning of every epoch
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    test_acc.reset_states()

    for (x_batch, y_batch) in dataset['train']:
        train_model(x_batch, y_batch)
        batch_train_ctr += 1

        template = 'Epoch {}, Batch {}, Loss: {}, Accuracy: {}'
        print(template.format(repeat, batch_train_ctr,
                              train_loss.result(),
                              train_acc.result() * 100))

        with train_summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss.result(), step=batch_train_ctr)
            tf.summary.scalar('train_accuracy', train_acc.result(), step=batch_train_ctr)

    for (x_batch, y_batch) in dataset['val']:
        test_model(x_batch, y_batch)
        batch_test_ctr += 1

        template = 'Epoch {}, Batch{}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(repeat, batch_test_ctr,
                              test_loss.result(),
                              test_acc.result() * 100))

        with test_summary_writer.as_default():
            tf.summary.scalar('test_loss', test_loss.result(), step=batch_test_ctr)
            tf.summary.scalar('test_accuracy', test_acc.result(), step=batch_test_ctr)

    show_predictions(dataset['val'], num=5)
    encoderDecoder.save_weights(os.getcwd()+"/weights/103layers_"+str(repeat+1)+"/")

