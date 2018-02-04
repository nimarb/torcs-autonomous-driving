# TORCS autonomous driving
This repository contains source code to train and drive a car in TORCS by itself.

## Dependencies
[TORCS-ROS](https://github.com/fmirus/torcs_ros), [TORCS](https://github.com/fmirus/torcs-1.3.7), [ROS Kinetic](http://wiki.ros.org/kinetic/Installation)  
python2: nengo, keras, tensorflow, numpy, opencv    
python3: keras, tensorflow, numpy, opencv  

## How to run
 1. start torcs: `torcs`
 2. configure the desired track, choose `scr_server` as driver
 3. run: `roslaunch torcs_ros_bringup torcs_ros.launch rviz:=false driver:=false`
 4. go into the nengo_controller folder and run: `python2 controller.py`

## Structure 
The `nengo_controller` folder contains the code needed to drive the car based on all the given sensor values.
The folder `src/collect_img_sensor_data` contains a ROS node to collect training data for the DNN.
The folder `src/train-deep-neural-network` contians code to train a deep neural network to infer angle and car displacement from a driver's view input image.

    ├── final-presentation-complete
    │   └── Bilder
    ├── nengo_controller
    │   ├── data
    │   │   └── processed_data
    │   └── nengo_ros
    ├── report
    │   ├── attachments
    │   └── paper
    └── src
        ├── collect_img_sensor_data
        │   ├── data-aalborg-2laps-640x480
        │   ├── data-alpine_1-2laps-640x480
        │   ├── data-alpine_2-2laps-640x480
        │   ├── data-brondehach-2laps-640x480
        │   ├── data-cg_speedway_1-2laps-640x480
        │   ├── data-cg_track_2-2laps-640x480
        │   ├── data-cg_track_3-2laps-640x480
        │   ├── data-cg_track_3-2laps-640x480-1sthood
        │   ├── data-cg_track_3-2laps-640x480-3rdclose
        │   ├── data-cg_track_3-2laps-640x480-3rdfar
        │   ├── data-corkscrew-2laps-640x480
        │   ├── data-e_road-2laps-640x480
        │   ├── data-etrack_1-2laps-640x480
        │   ├── data-etrack_2-2laps-640x480
        │   ├── data-etrack_3-2laps-640x480
        │   ├── data-etrack_4-2laps-640x480
        │   ├── data-etrack_6-2laps-640x480
        │   ├── data-forza-2laps-640x480
        │   ├── data-olethros_road_1-2laps-640x480
        │   ├── data-ruudskogen-2laps-640x480
        │   ├── data-street_1-2laps-640x480
        │   ├── data-wheel_1-2laps-640x480
        │   ├── data-wheel_2-2laps-640x480
        │   ├── launch
        │   └── src
        └── train-deep-neural-network
