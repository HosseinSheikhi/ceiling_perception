/**
 * Created by hossein on 2021-04-16.
 * Ceiling Perception Class
 * It subscribes to the segmented images of overhead cameras
 * and publishes an occupancy grid map using Static State Binary Base Filter
 * Assumptions:
 *      There is no overlap between FoV of cameras (minor overlaps will not lead to problem)
 *      All the num_overhead_cameras_ are placed somehow that cover a big rectangular FoV
 *
 *
 *  What I learned during developing:
 *  It is the different of assets and exceptions which is nicely explained in microsoft doc
 *
 *  Exceptions and asserts are two distinct mechanisms for detecting run-time errors in a program.
 *  1. Use assert statements to test for conditions during development that *should never be true if all your code is correct*.
 *  There's no point in handling such an error by using an exception, because the error indicates that something in the code has to be fixed.
 *  It doesn't represent a condition that the program has to recover from at run time.
 *  An assert stops execution at the statement so that you can inspect the program state in the debugger.
 *  An exception continues execution from the first appropriate catch handler.
 *  *Use exceptions to check error conditions that might occur at run time even if your code is correct, for example, "file not found" or "out of memory."*
 *  Exceptions can handle these conditions, even if the recovery just outputs a message to a log and ends the program.
 *  *Always check arguments to public functions by using exceptions. Even if your function is error-free, you might not have complete control over arguments that a user might pass to it.*
 *  Use asserts to check for errors that should never occur. Use exceptions to check for errors that might occur, for example, errors in input validation on parameters of public functions.
 *
 *  However, exception specifications proved problematic in practice, and are deprecated in the C++11 draft standard.
 *  void my_fun() throw(){} is deprecated
 *
 *  @see [simple example in multithread subscriers](https://github.com/ros2/examples/blob/master/rclcpp/executors/multithreaded_executor/multithreaded_executor.cpp)
 *
 */

#ifndef CEILING_PERCEPTION__CEILING_PERCEPTION_H_
#define CEILING_PERCEPTION__CEILING_PERCEPTION_H_
#include "ceiling_perception/overhead_camera_manager.h"
#include "custom_roi_srv/srv/roi.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "rclcpp/parameter_events_filter.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_ros/transform_broadcaster.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "cv_bridge/cv_bridge.h"
#include "opencv4/opencv2/highgui.hpp"
#include "opencv4/opencv2/opencv.hpp"
#include "memory.h"
#include "stdexcept"
#include <algorithm>
#include "chrono"
#include "cmath"
#include "random"



namespace ceiling_perception {
static constexpr signed char LETHAL_OBSTACLE = 100;
static constexpr signed char FREE_SPACE = 0;
class CeilingPerception : public rclcpp::Node {
public:
  explicit CeilingPerception(rclcpp::NodeOptions options);
private:

  /// a vector of camera manager (each subscribed frame will be assigned to one of this)
  std::vector<std::shared_ptr<OverheadCameraManager>> overhead_cameras_;
  /**
   * a vector of subscribers to overhead segmented images (these images are input to this class)
   * Note that because segmented frames are being published in a low frequency we do not need to assign subscribers in different threads
   */
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> camera_subs_;
  ///callback group for segmented images subscribers
  rclcpp::CallbackGroup::SharedPtr callback_group_subscriber_;

  /// timer to calculate and publish the map by an specific frequency
  rclcpp::TimerBase::SharedPtr map_timer_;

  ///callback group for map_timer
  rclcpp::CallbackGroup::SharedPtr callback_group_map_timer_;

  /// an occupancy grid map publisher
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_publisher_;

  /// To broadcast a tf
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_br_;


  /**
  * A callback for the map_timer_ which is called by a desired frequency
  * it calculate the posterior map by getting access to the prior map and new map
  * and finally publishes the map
  */
  void map_timer_callback();

  /**
   * publish a static tf just to be able to see ceiling_perception/map in RVIZ
   */
  void publish_static_tf();

  /**
   * calculates the logs odd of inverse sensor model based on current observations
   * @return logs odd of inverse sensor model
   */
  std::vector<float> inverse_sensor_model();


   /**
    * @param prob a vector which each element shows probabilities
    * @return converts the probabilities to log odds
    */
  inline std::vector<float> prob_to_logodds(std::vector<float> prob){
    std::vector<float> lo;
    lo.resize(prob.size());
    unsigned int index=0;
    for(auto & p : prob) {
      if (p <= 0.0000)
        p = 0.05 ;
      if (p >= 1.0)
        p = 0.95;
      lo[index++] = std::log(p / (1 - p));
    }
    return lo;
  }

  /**
   *
   * @param logg_odds a vector that each elements are log odds
   * @return converts the log odds to probabilities
   */
  inline std::vector<float> logodds_to_prob(std::vector<float> logg_odds){
    std::vector<float> prob;
    prob.resize(logg_odds.size());
    unsigned int index =0;
    for(auto const & lo : logg_odds)
      prob[index++] = 1 - (1/(1+std::exp(lo)));

    return prob;
  }


  /**
   * This function is a copy of costmap2d::world_to_map. converts the world coordinates to the map coordinate based on map origin and resolution.   Note: lowe-left is map's origin
   * @param wx[in], wy[in] world coordinates
   * @param mx[out], my[out] map coordinates
   * @return map coordinates corresponding to world coordinate
   */
  bool world_to_map(double wx, double wy, unsigned int & mx, unsigned int & my) const;

  /**
  * Given two map coordinates... compute the associated index - copy of costmap2d::get_index
  * @param mx The x coordinate
  * @param my The y coordinate
  * @return The associated index
  */
  inline unsigned int get_index(unsigned int mx, unsigned int my) const
  {
    return my * map_size_x_ + mx;
  }

  /**
   * Finds the min element across index column in the box vector
   * @param box a vector of vectors which are field by the coordinated of frames ROI
   * @param index shows which index of std::vector<double> have to be search to find the min
   *               e.g. if index=0 will return the minimum in the firs column (which is x_min)
   * @return min value in box across index element
   */
  double boxMin(const std::vector<std::vector<double>>& box, int index){

    std::vector<double> temp;
    for(auto const& vec : box)
      temp.push_back(vec[index]);

    return *std::min_element(temp.begin(), temp.end());
  }

  /**
   * Finds the max element across index column in the box vector
   * @param box a vector of vectors which are field by the coordinated of frames ROI
   * @param index shows which index of std::vector<double> have to be search to find the max
   *        e.g. if index=2 will return the maximum in the firs column (which is x_max)
   * @return max value in box across index element
   */
  double boxMax(const std::vector<std::vector<double>>& box, int index){
    std::vector<double> temp;
    for(auto const& vec : box)
      temp.push_back(vec[index]);

    return *std::max_element(temp.begin(), temp.end());
  }

  /**
   * calculates the perception_layer region of interest and desired map size as well
   */
  void calculate_roi();

  /**
   * posterior at time t would be prior at time t+1
   * we use a saturated prior update otherwise is wont show the changes in environment
   * @param posterior posterior map log odds
   */
  void update_prior(std::vector<float> posterior){
    for(unsigned int i=0;i< posterior.size();i++) {
      if (posterior[i] > 0)
        prior_map_log_odds_[i] = std::min(
            log_odds_saturation_,
            posterior[i]); // posterior at time t will be prior at next time
      else {
        prior_map_log_odds_[i] = std::max(
            -log_odds_saturation_,
            posterior[i]); // posterior at time t will be prior at next time
      }
    }
  }

  std::vector<float> prior_map_log_odds_;
  double roi_min_x_{0}, roi_min_y_{0}, roi_max_x_{0}, roi_max_y_{0}; ///< to store the ceiling_perception ROI based on frames coverage
  unsigned int map_size_x_, map_size_y_; ///< desired size of x and y in meter based on frames coverage, must divided by resolution
  double origin_x_, origin_y_; ///< stores the map's origin in world coordinate


  // ROS2 Parameters
  void get_parameters(); ///< reads parameters from launch file
  bool enabled_; ///< whether if this node is enabled
  bool publish_tf_ = false; ///< whether if static tf must be published
  int num_overhead_cameras_; ///< Number of overhead cameras will be fed to this class to get a map
  std::vector<std::vector<float>> camera_poses_; ///< a vector of (x,y,z) coordinates of all cameras
  std::vector<std::string> overhead_topics_; ///< a vector of topic names that segmented images are published on
  double resolution_; ///< map resolution, must be as same as SLAM resolution
  /**
   * use to saturate prior log odds
   * set zero to do not use prior (map will be calculated just using current measurements)
   * the higher the saturation the previous measurements will accumulate more, so slower response to changes in environments
   */
  float log_odds_saturation_;
};
}

#endif // CEILING_PERCEPTION__CEILING_PERCEPTION_H_
