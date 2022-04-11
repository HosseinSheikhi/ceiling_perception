from tkinter import N
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from UNET.VGG16 import vgg16_train, vgg16_inference
# from ceiling_segmentation.DenseNet.models import densenet_train, densenet_inference
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import message_filters
import numpy as np

class CeilingSegmentation(Node):
    def __init__(self):
        super(CeilingSegmentation, self).__init__('ceiling_segmentation')
        self.show_images = False
        self.mode = None
        self.method = None
        self.frequency = None
        self.overhead_topics = None
        self.image_publisher = None
        self.image_subscriber = None

        self.batch_size = None
        self.image_size = None
        self.buffer_size = None
        self.epoch = None
        self.seed = None
        self.num_channels = None
        self.num_classes = None
        self.data_address = None
        self.weight_address = None
        self.parameters()

    def parameters(self):
        """
        Define ROS2 parameters and read from config file
        :return: None
        """
        param_mode_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                    description='This parameter is used to determine this node must be in train or inference mode.')
        self.declare_parameter('mode', "inference", param_mode_descriptor)

        param_method_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                                      description='This parameter is used to determine the method that must be run in this node. Could be VGG16 or DenseNet')
        self.declare_parameter('method', "VGG16", param_method_descriptor)

        param_frequency_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE,
                                                         description='This parameter is used to determine the frequency of publishing segmented images')
        self.declare_parameter('frequency', 1.0, param_frequency_descriptor)

        param_overhead_topics_descriptor = ParameterDescriptor(type=ParameterType.PARAMETER_STRING_ARRAY,
                                                               description='This parameter is used to find topics that overhead cameras are publishing on. (must be subscribed by this package)')
        self.declare_parameter('overhead_topics', ["/overhead_cam_1/camera/image_raw"], param_overhead_topics_descriptor)

        self.declare_parameter('image_size', 224)
        self.declare_parameter('num_channels', 3)
        self.declare_parameter('num_classes', 2)
        self.declare_parameter('batch_size', 2)
        self.declare_parameter('epoch', 2)
        self.declare_parameter('buffer_size', 4)
        self.declare_parameter('seed', 4)
        self.declare_parameter('data_address',"/home/hossein/synthesisData_Jan2021")
        self.declare_parameter('weight_address', "/home/hossein/nav2_ws/src/ceiling_perception/ceiling_segmentation/weights/WithoutBN/NaiveLoss2/")


        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        if self.mode != "train" or self.mode != "inference":
            self.get_logger().error("Mode in config file must be either train or inference")
        self.method = self.get_parameter('method').get_parameter_value().string_value
        if self.method != "VGG16" or self.method != "DenseNet":
            self.get_logger().error("Method in config file must be either VGG16 or DenseNet")
        self.frequency = self.get_parameter('frequency').get_parameter_value().double_value
        self.overhead_topics = self.get_parameter('overhead_topics').get_parameter_value().string_array_value

        self.image_size = self.get_parameter('image_size').get_parameter_value().integer_value
        self.num_channels = self.get_parameter('num_channels').get_parameter_value().integer_value
        self.num_classes = self.get_parameter('num_classes').get_parameter_value().integer_value
        self.batch_size = self.get_parameter('batch_size').get_parameter_value().integer_value
        self.epoch = self.get_parameter('epoch').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.seed = self.get_parameter('seed').get_parameter_value().integer_value
        self.data_address = self.get_parameter('data_address').get_parameter_value().string_value
        self.weight_address = self.get_parameter('weight_address').get_parameter_value().string_value

    def inference(self, model):
        self.model = model
        self.image_publisher = [self.create_publisher(Image,
                                                      "autonomous_robot/image_segmentation/image_"+str(i+1),
                                                      qos_profile_system_default)
                                for i in range(len(self.overhead_topics))]

        self.image_subscriber = [
            message_filters.Subscriber(self, Image, topic,
                                       qos_profile=qos_profile_sensor_data)
            for topic in self.overhead_topics]


        self.segmented_images = []
        # define the policy for message filtering
        syn = message_filters.ApproximateTimeSynchronizer(self.image_subscriber, 1, 0.1)
        # message_filter::subscriber will start subscribing upon being registered
        syn.registerCallback(self.image_callback)
        self.timer = self.create_timer(1/self.frequency, self.publish_segmented_images)

    def publish_segmented_images(self):
        self.get_logger().info('Publishing images ...')
        for idx, publisher in enumerate(self.image_publisher):
            publisher.publish(CvBridge().cv2_to_imgmsg(self.segmented_images[idx]))
    
    def image_callback(self, *images):  
        self.get_logger().info('Images received')
        cv_images = []
        for image in images:
            cv_images.append(CvBridge().imgmsg_to_cv2(image, desired_encoding='bgr8'))

        self.segmented_images.clear()
        predicted_images = self.model.inference(*cv_images)
        for i in range(len(self.overhead_topics)):
            self.segmented_images.append(predicted_images[i, :, :].astype(np.float32) * 255.0)  # 2 (224,224)

        if self.show_images:
            for i, cv_image in enumerate(cv_images):
                cv2.imshow("image" + str(i + 1), cv_image)
            cv2.waitKey(1)

            for i, segmented_image in enumerate(self.segmented_images):
                cv2.imshow("segmented image" + str(i + 1), segmented_image)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    ceiling_segmentation = CeilingSegmentation()

    if ceiling_segmentation.method == "VGG16":
        if ceiling_segmentation.mode == "train":
            vgg = vgg16_train.VGG16Train(
                    ceiling_segmentation.image_size,
                    ceiling_segmentation.num_channels,
                    ceiling_segmentation.num_classes,
                    ceiling_segmentation.batch_size,
                    ceiling_segmentation.buffer_size,
                    ceiling_segmentation.epoch,
                    ceiling_segmentation.seed,
                    ceiling_segmentation.data_address
                    )
            vgg.load_data()
            vgg.build_model()
            vgg.train_procedure()

        elif ceiling_segmentation.mode == "inference":
            vgg = vgg16_inference.VGG16Inference(
                ceiling_segmentation.num_classes,
                ceiling_segmentation.image_size,
                ceiling_segmentation.weight_address
            )
            vgg.build_model()
            # vgg.inference_from_file("/home/hossein/first_overhead_cam.jpg")
            ceiling_segmentation.inference(vgg)
            rclpy.spin(ceiling_segmentation)
            rclpy.shutdown()
    elif ceiling_segmentation.method == "DenseNet":
        if ceiling_segmentation.mode == "train":
            pass
        elif ceiling_segmentation.mode == "inference":
            pass

if __name__ == "__main__":
    main()