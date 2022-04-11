//
// Created by hossein on 2021-04-16.
//

#include "ceiling_perception/overhead_camera_manager.h"

ceiling_perception::OverheadCameraManager::OverheadCameraManager(std::string name,
                                            double pose_x,
                                            double pose_y,
                                            double pose_z):
    name_(name), pose_x_(pose_x), pose_y_(pose_y), pose_z_(pose_z)
{
  image_height_ = 480;
  image_width_ = 640;
  focal_x_ = focal_y_ = 381.362;
  x_0_ = 320.5;
  y_0_ = 240.5;
  occupied_prob_.resize(image_height_, std::vector<float>(image_width_));
}

ceiling_perception::OverheadCameraManager::OverheadCameraManager(std::string name,
                                            double pose_x,
                                            double pose_y,
                                            double pose_z,
                                            unsigned int image_height,
                                            unsigned int image_width):
    name_(name), pose_x_(pose_x), pose_y_(pose_y), pose_z_(pose_z),
    image_height_(image_height), image_width_(image_width)
{
  focal_x_ = focal_y_ = 381.362;
  x_0_ = 320.5;
  y_0_ = 240.5;
  occupied_prob_.resize(image_height_, std::vector<float>(image_width_));

}
ceiling_perception::OverheadCameraManager::OverheadCameraManager(std::string name,
                                            double pose_x,
                                            double pose_y,
                                            double pose_z,
                                            unsigned int image_height,
                                            unsigned int image_width,
                                            double focal_x,
                                            double focal_y,
                                            double x_0,
                                            double y_0):
    name_(name), pose_x_(pose_x), pose_y_(pose_y), pose_z_(pose_z),
    image_height_(image_height), image_width_(image_width),
    focal_x_(focal_x),
    focal_y_(focal_y),
    x_0_(x_0),
    y_0_(y_0)
{
  occupied_prob_.resize(image_height_, std::vector<float>(image_width_));
}

bool ceiling_perception::OverheadCameraManager::pixelToWorld(unsigned int x_pixel,
                                                               unsigned int y_pixel,
                                                               double &x_world,
                                                               double &y_world)
{
  x_world = (static_cast<double>(x_pixel) - x_0_)*pose_z_/focal_x_ + pose_x_;
  y_world = -(static_cast<double>(y_pixel) - y_0_)*pose_z_/focal_y_ + pose_y_;
  if(x_world>= world_x_min_ && x_world <= world_x_max_ && y_world >= world_y_min_ && y_world <= world_y_max_)
    return true;
  else
    return false;
}

bool ceiling_perception::OverheadCameraManager::OverheadCameraManager::worldToPixel(double x_world,
                                                               double y_world,
                                                               unsigned int &x_pixel,
                                                               unsigned int &y_pixel)
{
  x_pixel = static_cast<unsigned int>((x_world-pose_x_)*focal_x_/pose_z_+x_0_);
  y_pixel = static_cast<unsigned int>(-(y_world-pose_y_)*focal_y_/pose_z_+y_0_);
  if(x_pixel <= image_width_ && y_pixel<=image_height_)
    return true;
  else
    return false;
}

void ceiling_perception::OverheadCameraManager::image_cb(sensor_msgs::msg::Image::SharedPtr image) {
  cv_bridge::CvImagePtr cv_image_ptr = cv_bridge::toCvCopy(image);
  segmented_image_ = cv_image_ptr->image;
  cv::resize(segmented_image_, segmented_image_,
             cv::Size(image_width_, image_height_));
  #if DEBUG
  cv::imshow(name_, segmented_image_);
  cv::waitKey(1);
  std::cout << "Frame " << this->name_ << " received." << std::endl;
  #endif
  update_ = true; /// instance is update when image is being subscribed
}

void ceiling_perception::OverheadCameraManager::calculate_frame_occ_prob(){
  std::normal_distribution<float> distribution(0.0,0.05);

  occupied_prob_.clear();
  occupied_prob_.resize(image_height_, std::vector<float>(image_width_));

  // to check the probability of a pixel  being occupied we loot at a neighborhood of that pixel
  int neighbour_size = 5;
  cv::Mat temp_img;

  // field frame is copied to a temp_img mat because maybe callback function be called during this function and change th fields value
  segmented_image_.copyTo(temp_img);
  if(temp_img.empty() || temp_img.cols!= static_cast<int>(image_width_) ||
      temp_img.rows != static_cast<int>(image_height_) ){
    return ;
  }
  // segmented image is in Black(Free) and White(Occluded)
  int white_pixels_counter=0;
  int total_pixels_counter=0;

  for( int x_pixel = 0 ; x_pixel< static_cast<int>(image_width_); x_pixel++)
    for( int y_pixel = 0; y_pixel< static_cast<int>(image_height_); y_pixel++) {
      white_pixels_counter=0;
      total_pixels_counter=0;
      for (int i = std::max(x_pixel - neighbour_size, 0); i <= std::min(x_pixel + neighbour_size, static_cast<int>(image_width_) - 1);i++)
        for (int j = std::max(y_pixel - neighbour_size, 0); j <= std::min(y_pixel + neighbour_size, static_cast<int>(image_height_) - 1);j++) {
          if (int(temp_img.at<float>(j, i)) > 60) {
            white_pixels_counter++;
          }
          total_pixels_counter++;
        }

      occupied_prob_[y_pixel][x_pixel] =(static_cast<float>(white_pixels_counter) / static_cast<float>(total_pixels_counter))+ distribution(generator_);
    }
  // normalize_vector the occupied_prob_ between [0,1]
  normalize_vector();
}

void ceiling_perception::OverheadCameraManager::normalize_vector(){
  std::vector<float> temp;
  for(auto & row : occupied_prob_){ // iterate over each row vector and find its max and min
    temp.emplace_back(*std::max_element(row.begin(), row.end()));
    temp.emplace_back(*std::min_element(row.begin(), row.end()));
  }


  auto max = *std::max_element(temp.begin(), temp.end());
  auto min = *std::min_element(temp.begin(), temp.end());
  auto diff = max-min;

  for(auto & row : occupied_prob_)
    for(auto & element: row)
      element =  (element-min)/diff;
}


void ceiling_perception::OverheadCameraManager::world_FOV(double &min_x,
                                                          double &min_y,
                                                          double &max_x,
                                                          double &max_y) {
  pixelToWorld(static_cast<unsigned int>(0),image_height_,min_x, min_y);
  pixelToWorld(image_width_,static_cast<unsigned int>(0), max_x, max_y);
  world_x_min_ = min_x;
  world_y_min_ = min_y;
  world_x_max_ = max_x;
  world_y_max_ = max_y;
}

