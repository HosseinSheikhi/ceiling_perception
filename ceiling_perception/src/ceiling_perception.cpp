//
// Created by hossein on 2021-04-16.
//

#include "ceiling_perception/ceiling_perception.h"
#include "ceiling_perception/array_parser.h"
#include "rclcpp/qos.hpp"

using namespace std::chrono_literals;

ceiling_perception::CeilingPerception::CeilingPerception(rclcpp::NodeOptions options)
    :Node("ceiling_perception", options) {
  /// declare parameters must be read from launch file
  this->declare_parameter<bool>("enabled", true);
  this->declare_parameter<float>("log_odds_saturation", 10.0);
  this->declare_parameter<int>("num_overhead_cameras", 1);
  this->declare_parameter<double>("resolution", 0.05);
  this->declare_parameter<std::string>("camera_poses", "[[0, 0, 3.5]");
  this->declare_parameter<std::vector<std::string>>("overhead_topics", std::vector<std::string>());

  /// read declared parameters, we need them from now on
  try {
    get_parameters();
  }
  catch(std::invalid_argument &e) {
    // nothing to free and care about so just rethrow
    throw;
  }

  map_publisher_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("map",rclcpp::SystemDefaultsQoS());

  tf_br_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
  /**
   * define the callback group
   * They don't really do much on their own, but they have to exist in order to
   * assign callbacks to them. They're also what the executor looks for when trying to run multiple threads
   */
  callback_group_subscriber_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  callback_group_map_timer_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  /**
   * Each of these callback groups is basically a thread
   * Everything assigned to one of them gets bundled into the same thread
   */
  auto sub_opt = rclcpp::SubscriptionOptions();
  sub_opt.callback_group = callback_group_subscriber_;


  /// define timer and bind it to its subscriber
  map_timer_ = this->create_wall_timer(
      5s, std::bind(&CeilingPerception::map_timer_callback, this),
      callback_group_map_timer_);

  /// define subscribers to the segmented topics and assign each of them to a camera manager
  for (int cam_index = 0; cam_index < num_overhead_cameras_; cam_index++) {
    overhead_cameras_.emplace_back(std::make_shared<OverheadCameraManager>(
        "overhead_cam_" + std::to_string(cam_index + 1),
        camera_poses_[cam_index][0], camera_poses_[cam_index][1],
        camera_poses_[cam_index][2]));
    camera_subs_.emplace_back(
        this->create_subscription<sensor_msgs::msg::Image>(
            overhead_topics_[cam_index], rclcpp::SystemDefaultsQoS(),
            std::bind(&OverheadCameraManager::OverheadCameraManager::image_cb,
                      overhead_cameras_[cam_index], std::placeholders::_1),
            sub_opt));
  }

    calculate_roi();

}

void ceiling_perception::CeilingPerception::map_timer_callback() {

#if DEBUG
  RCLCPP_INFO(this->get_logger(), "Ceiling Perception: Creating a map");
#endif

  auto inverse_sensor_prob = inverse_sensor_model(); // get inverse sensor probability
  auto inverse_sensor_log_odds = prob_to_logodds(inverse_sensor_prob); // get inverse sensor log odds

  std::vector<float> posterior_map_log_odds(prior_map_log_odds_.size());
  for(unsigned int i=0;i< prior_map_log_odds_.size();i++)
    posterior_map_log_odds[i] = prior_map_log_odds_[i] + inverse_sensor_log_odds[i]; //sum inverse sensor log odds by prior map to get posterior log odds

  auto posterior_map_prob = logodds_to_prob(posterior_map_log_odds); // convert posterior log odds to probabilities

  std::vector<int8_t> map(posterior_map_prob.size());
  for(unsigned int i = 0; i< posterior_map_prob.size(); i++)
    map[i] = static_cast<int8_t>(round(posterior_map_prob[i]))*100;

  // create a map message and fill the fields
  nav_msgs::msg::OccupancyGrid ceiling_map;
  ceiling_map.header.frame_id = "ceiling_perception";
  ceiling_map.header.stamp = this->now();

  ceiling_map.info.origin.position.set__x(roi_min_x_);
  ceiling_map.info.origin.position.set__y(roi_min_y_);
  ceiling_map.info.resolution = static_cast<float>(resolution_);
  ceiling_map.info.height = map_size_y_;
  ceiling_map.info.width = map_size_x_;

  ceiling_map.data= map;

  //publish the ceiling_map
  map_publisher_->publish(ceiling_map);

  update_prior(posterior_map_log_odds);

  if(publish_tf_)
    publish_static_tf();

#if DEBUG
  RCLCPP_INFO(this->get_logger(), "Ceiling Perception: map published");
#endif
}

std::vector<float> ceiling_perception::CeilingPerception::inverse_sensor_model(){
#if DEBUG
  RCLCPP_INFO(this->get_logger(), "Ceiling Perception: creating a inverse_sensor_prob based on inverse sensor model (Just current measurements)");
#endif

  std::vector<float> inverse_sensor_prob; // keeps track of p(m_{i} | z_{t}, x_{t})
  inverse_sensor_prob.resize(map_size_x_ * map_size_y_,0.0); // this should not be initialized by 0.5

  for (auto const & frame : overhead_cameras_) { //iterate on each camera's frame
    if (frame->isUpdate()) { // if camera manager is receiving new frames
      frame->calculate_frame_occ_prob();
      auto occ_prob = frame->get_frame_occ_prob(); // get the probability of each pixels in a frame being occupied or free
        for (unsigned int x_pixel = 0; x_pixel < 640; x_pixel++) {  // iterate over current frame's pixels in row
          for (unsigned int y_pixel = 0; y_pixel < 480; y_pixel++) { // iterate over current frame's pixels in cols
            double x_world, y_world;
            if (frame->pixelToWorld(x_pixel, y_pixel, x_world, y_world)) { // convert current pixel to the world coordinate
              unsigned int x_map, y_map;
              if (world_to_map(x_world, y_world, x_map,y_map)) { // convert world coordinates to the inverse_sensor_prob coordinates
                unsigned int index = get_index(x_map, y_map); // get index of the inverse_sensor_prob coordinate
                if(inverse_sensor_prob[index] == 0) // this is first time we meet this cell
                  inverse_sensor_prob[index] = occ_prob[y_pixel][x_pixel];
                else{ // we might meet a cell more than once
                  inverse_sensor_prob[index] = (inverse_sensor_prob[index]+ occ_prob[y_pixel][x_pixel])/2;
                }
              }
            } else {
              RCLCPP_WARN(this->get_logger(),
                          "Ceiling Perception: pixelToWorld was not successful");
            }
          }
        }
      }
    }

  return inverse_sensor_prob;
}


void ceiling_perception::CeilingPerception::publish_static_tf(){
  geometry_msgs::msg::TransformStamped  transform_stamped;

  transform_stamped.header.stamp = this->now();
  transform_stamped.header.frame_id = "ceiling_perception";
  transform_stamped.child_frame_id = "odom";

  tf_br_->sendTransform(transform_stamped);

#if DEBUG
  RCLCPP_INFO(this->get_logger(), "Ceiling Perception: static tf published");
#endif
}



bool ceiling_perception::CeilingPerception::world_to_map(double wx, double wy, unsigned int & mx, unsigned int & my) const
{
  if (wx < origin_x_ || wy < origin_y_) {
    return false;
  }

  mx = static_cast<int>((wx - origin_x_) / resolution_);
  my = static_cast<int>((wy - origin_y_) / resolution_);

  if (mx < map_size_x_ && my < map_size_y_) {
    return true;
  }

  return false;
}

void ceiling_perception::CeilingPerception::calculate_roi(){
  std::vector<std::vector<double>> FOVBoxes;
  double x1_min, y1_min, x1_max, y1_max;

  for (const auto & cam_manager : overhead_cameras_){
    cam_manager->world_FOV(x1_min, y1_min, x1_max, y1_max);
    FOVBoxes.emplace_back(std::vector<double>{x1_min, y1_min, x1_max, y1_max});
  }

  roi_min_x_ = boxMin(FOVBoxes, 0);
  roi_min_y_ = boxMin(FOVBoxes, 1);

  roi_max_x_ = boxMax(FOVBoxes, 2);
  roi_max_y_ = boxMax(FOVBoxes, 3);
  map_size_x_ = static_cast<unsigned int>(fabs(roi_max_x_ - roi_min_x_)/resolution_);
  map_size_y_ = static_cast<unsigned int>(fabs(roi_max_y_ - roi_min_y_)/resolution_);

  origin_x_ = roi_min_x_;
  origin_y_ = roi_min_y_;

  prior_map_log_odds_.resize(map_size_x_*map_size_y_,0.0);

#if DEBUG
    RCLCPP_INFO(this->get_logger(), "Ceiling Perception: Desired ROI is min(%f,%f) - max(%f,%f)",roi_min_x_,roi_min_y_, roi_max_x_, roi_max_y_ );
  #endif
}


void ceiling_perception::CeilingPerception::get_parameters() {

  this->get_parameter("enabled", enabled_);
  this->get_parameter("publish_tf", publish_tf_);
  this->get_parameter("log_odds_saturation", log_odds_saturation_);
  this->get_parameter("num_overhead_cameras", num_overhead_cameras_);
  this->get_parameter("resolution", resolution_);
  this->get_parameter("overhead_topics", overhead_topics_);

  std::string camera_poses_str;
  this->get_parameter("camera_poses", camera_poses_str);
  std::string error;
  camera_poses_ = parseVVF(camera_poses_str, error);
  if (!error.empty())
    throw std::invalid_argument("CeilingPerception: error in parsing camera poses " );

  if(overhead_topics_.size() != static_cast<unsigned long>(num_overhead_cameras_))
    throw std::invalid_argument("CeilingPerception: number of overhead cameras doesn't match overhead topics");

  if(camera_poses_.size() != static_cast<unsigned long>(num_overhead_cameras_))
    throw std::invalid_argument("CeilingPerception: number of overhead cameras doesn't match cameras positions");

}