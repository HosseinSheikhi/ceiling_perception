from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import EnvironmentVariable
import os
import launch_ros.actions
import pathlib

parameters_file_name = 'ceiling_perception_params.yaml'

def generate_launch_description():
    parameters_file_path = str(pathlib.Path(__file__).parents[1]) # get current path and go one level up
    parameters_file_path += '/config/' + parameters_file_name
    return LaunchDescription([
        Node(
            package='ceiling_perception',
            namespace='ceiling_perception',
            executable='ceiling_perception_start',
            output='screen',
               parameters=[
                              parameters_file_path
                          ],
        ),
    ])