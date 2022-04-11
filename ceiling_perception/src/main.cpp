//
// Created by hossein on 2021-04-17.
//

#include "../include/ceiling_perception/ceiling_perception.h"
#include "rclcpp/rclcpp.hpp"
#include <iostream>

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  std::shared_ptr<ceiling_perception::CeilingPerception> ceiling_perception;
  try {
    ceiling_perception = std::make_shared<ceiling_perception::CeilingPerception>(node_options);
  }
  catch(std::invalid_argument &e) {
    std::cerr<<"Exception in creating Ceiling Perception: "<<e.what()<<std::endl;
    return -1;
  }
  /// You MUST use the MultiThreadedExecutor to use, well, multiple threads
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(ceiling_perception);
  executor.spin();
  rclcpp::shutdown();

  return 0;
}