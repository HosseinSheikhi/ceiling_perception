# ceiling_perception

# Setup realsense
`sudo chmod 777 /dev/video*`
`ros2 launch realsense2_camera rs_launch.py`

Probably you will get a white -unbalanced- image to solve it you have to reconfigure the dynamic parameters:
`ros2 run rqt_reconfigure rqt_reconfigure`
change the rgb camera auto exposure rot bottom to zero

camera calibration parameters:
```
height: 480
width: 640
distortion_model: plumb_bob
d:
- -0.05795379728078842
- 0.06353622674942017
- -0.00044567903387360275
- 6.581264460692182e-05
- -0.01993693597614765
k:
- 381.52740478515625
- 0.0
- 317.77301025390625
- 0.0
- 381.1669921875
- 250.42886352539062
- 0.0
- 0.0
- 1.0
r:
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
- 0.0
- 0.0
- 1.0
p:
- 381.52740478515625
- 0.0
- 317.77301025390625
- 0.0
- 0.0
- 381.1669921875
- 250.42886352539062
- 0.0
- 0.0
- 0.0
- 1.0
- 0.0
```
