# Build
Build in debug and release mode (think default is debug)

    colcon build --packages-select ceiling_perception --cmake-args -DCMAKE_BUILD_TYPE=Release

    colcon build --packages-select ceiling_perception --cmake-args -DCMAKE_BUILD_TYPE=Debug

# Visualize in RVIZ
    set the fixed frame to ceiling_perception and set topic and update topic attributes to systemdefault
    generate an static TF:
        ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0  ceiling_perception odom


# ROS2 Multithreading
ROS 2 addresses thread-pool starvation by introducing Callback Groups, which:

    ● Group of things with callbacks, like Subscriptions and Timers
    ● Are completely optional constructs, hidden to users by default
    ● Works with both component style and the custom main style nodes.

Callbacks in Reentrant Callback Groups must be able to:

    ● run at the same time as themselves (reentrant)
    ● run at the same time as other callbacks in their group
    ● run at the same time as other callbacks in other groups

Whereas, callbacks in Mutually Exclusive Callback Groups:

    ● will not be run multiple times simultaneously (non-reentrant)
    ● will not be run at the same time as other callbacks in their group
    ● but must run at the same time as callbacks in other groups

Users can use Mutually Exclusive Callback Groups in order to ensure
callbacks which operate on shared resources (frames in subscribers and map are shared in our code) do not run at the same
time, without using locks. 

So I have not used mutex to get access to frames in map_timer callback.

# TODO
    publish TF