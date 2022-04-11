/**
 * Created by hossein on 2021-04-16.
 *
 * Overhead_camera_manager class
 *
 * it does:
 *  Each subscribed segmented image in Ceiling_perception class is assigned to
 *  an object of this class so we have a few utility functions for each image
 *
 * Note:
 * Cameras are placed some how that their lower left has min coordinates in world and upper right max coordinate in world
 * if following is a Top vies of a frame
 *                      -------------------------------(max_wx, max_wy)|
 *                      |                                              |
 *                      |                                              |
 *                      |                                              |
 *                      |                                              |
 *                      (min_xw, min_xy)-------------------------------|
 */

#ifndef CEILING_PERCEPTION__OVERHEAD_CAMERA_MANAGER_H_
#define CEILING_PERCEPTION__OVERHEAD_CAMERA_MANAGER_H_
#include "sensor_msgs/msg/image.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "cv_bridge/cv_bridge.h"
#include "iostream"
#include "random"
namespace ceiling_perception {
class OverheadCameraManager {
public:
  /**
   * @param name name of the assigned cv image e.g. overhead_1
   * @param pose_x,pose_y,pose_z (x,y,z) position of corresponding overhead camera
   */
  OverheadCameraManager(std::string name, double pose_x, double pose_y,
                          double pose_z);

  /**
   * @param name name of the assigned cv image e.g. overhead_1
   * @param pose_x,pose_y,pose_z (x,y,z) position of corresponding overhead camera
   * @param image_height, image_width size ( or height X width) of the image e.g. 640 X 480
   */
  OverheadCameraManager(std::string name, double pose_x, double pose_y,
                          double pose_z, unsigned int image_height,
                          unsigned int image_width);

  /**
   * @param name name of the assigned cv image e.g. overhead_1
   * @param pose_x,pose_y,pose_z (x,y,z) position of corresponding overhead camera
   * @param image_height, image_width height X width of the image e.g. 640 X 480
   * @param focal_x, focal_y, x_0, y_0 a camera calibration parameters
   */
  OverheadCameraManager(std::string name, double pose_x, double pose_y,
                          double pose_z, unsigned int image_height,
                          unsigned int image_width, double focal_x,
                          double focal_y, double x_0, double y_0);

  /**
   * @return True if a new frame is passed to this class otherwise false
   */
  inline bool isUpdate() const { return update_; }

  /**
   * @param update a boolean is passed to change the update_ statues (true if new frame is passed to this class)
   */
  inline void setUpdate(bool update) { update_ = update; }

  /**
   * converts the pixel coordinates to the world coordinates
   * @see [calculate world coordinates from frame coordinates](https://stackoverflow.com/questions/12007775/to-calculate-world-coordinates-from-screen-coordinates-with-opencv)
   * @param x_pixel[in], y_pixel[in] (x,y) pixel coordinates
   * @param x_world[out], y_world[out] (x,y) worlds coordinates corresponding to the input pixel
   * @return true if conversion was successful false otherwise
   */
  bool pixelToWorld(unsigned int x_pixel, unsigned int y_pixel, double &x_world,
                    double &y_world);

  /**
   * converts the world coordinate to the pixel coordinates
   * @param x_world[in], y_world[in] (x,y) world coordinate
   * @param x_pixel[out], y_pixel[out]  (x,y) pixel coordinate corresponding to the input pixel
   * @return true if conversion was successful false otherwise
   */
  bool worldToPixel(double x_world, double y_world, unsigned int &x_pixel,
                    unsigned int &y_pixel);

  /**
   * callback function for the ROS2 subscriber to the segmented image (subscriber is defined in Ceiling_perception class)
   * @param image subscribed image
   */
  void image_cb(sensor_msgs::msg::Image::SharedPtr image);

  /**
   * Normalize the given vector to 0.0-1.0 range
   * @param prob[in/out] vector that needs normalization
   */
  void normalize_vector();
   /**
    * calculates the probability of that pixel being occupied based on #num of occupied pixels in a neighborhood/#num of total pixels in a neighborhood
    * and fills the occupied_prob_ vector
    * @return
    */
  void calculate_frame_occ_prob();

   /**
    * @return the probability of each pixel being occupied
   */
  inline std::vector<std::vector<float>> get_frame_occ_prob() const{return occupied_prob_;}

  /**
   * @param px, py
   * @return probability of pixel (px, py) being occupied
   */
  inline float get_pixel_occ_prob(unsigned int px, unsigned int py) const{
    return occupied_prob_[px][py];
  }

  /**
   * calculates the rectangular area (in world coordinate) covered by this class's frame
   * @param min_x[out], min_y[out], max_x[out], max_y[out] rectangle coordinates covered (Field Of View) by this class's image
   */
  void world_FOV(double &min_x, double &min_y, double &max_x, double &max_y);

  /**
   * check if the inputs coordinates (in world) are within FoV of this class's image
   * @param wx, wy coordinates in world
   * @return True if input coordinates are withing FoV false otherwise
   */
  inline bool cover_world(double wx, double wy) {
    if (world_x_min_ < wx && world_y_min_ < wy && world_x_max_ > wx &&
        world_y_max_ > wy)
      return true;
    else
      return false;
  }

private:

  /// To add random noise to inverse sensor model
  std::default_random_engine generator_;

  std::string name_; ///< to store image name e.g. overhead_1
  double pose_x_, pose_y_,
      pose_z_; ///< to store camera position in cartesian coordinates
  unsigned int image_height_, image_width_; ///< to store image size
  double focal_x_, focal_y_, x_0_,
      y_0_; ///< camera calibration parameters. are needed in PoxelToWorl and vice versa
  cv::Mat segmented_image_; ///< will keep track of the assigned image frame
  double world_x_min_, world_y_min_, world_x_max_,
      world_y_max_; ///< to keep the cartesian coordinates of the area that is covered by the overhead camera
  bool update_{false}; ///< flag to check if a new frame is passed to this class
  std::vector<std::vector<float>> occupied_prob_; /// to store of probability of each cell being occupied. fills in calculate_frame_occ_prob()
};
}

#endif // CEILING_PERCEPTION__OVERHEAD_CAMERA_MANAGER_H_
