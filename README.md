# aichallenge_gym_ros

`aichallenge_gym_ros` is a project that adapts the [f1tenth_gym](https://github.com/f1tenth/f1tenth_gym) environment for use within the AI Challenge competition context. This repository provides a bridge between the ROS 2 environment and the `aic_gym` simulation, allowing for control and sensing of a simulated vehicle.

## Purpose

The primary goal of `aichallenge_gym` is to enable participants of the AI Challenge to develop and test their autonomous driving algorithms using a familiar Gym-compatible simulation environment, while integrating with the ROS 2 ecosystem for communication and control.

## Parameters

The following parameters can be declared and set for the `GymBridge` node:

| Parameter Name            | Description                                                               |
| :------------------------ | :------------------------------------------------------------------------ |
| `ego_scan_topic`          | ROS topic for publishing ego vehicle's laser scan data.                   |
| `scan_distance_to_base_link` | Distance from the laser scanner to the base link of the vehicle.          |
| `scan_fov`                | Field of view of the laser scanner in radians.                            |
| `scan_beams`              | Number of beams in the laser scan.                                        |
| `map_path`                | Path to the map file used in the Gym environment.                         |
| `map_img_ext`             | Image extension of the map file (e.g., `.png`, `.yaml`).                  |
| `sx`                      | Initial X-coordinate of the ego vehicle's pose.                           |
| `sy`                      | Initial Y-coordinate of the ego vehicle's pose.                           |
| `stheta`                  | Initial yaw (theta) of the ego vehicle's pose in radians.                 |
| `kb_teleop`               | Boolean flag to enable keyboard teleoperation (subscribes to `/cmd_vel`). |
| `mass`                    | Mass of the vehicle in kg.                                                |
| `length`                  | Length of the vehicle in meters.                                          |
| `width`                   | Width of the vehicle in meters.                                           |
| `lf`                      | Distance from the center of gravity to the front axle in meters.          |
| `lr`                      | Distance from the center of gravity to the rear axle in meters.           |
| `s_max`                   | Maximum steering angle in radians.                                        |
| `s_min`                   | Minimum steering angle in radians.                                        |
| `a_max`                   | Maximum acceleration in m/s^2.                                            |
| `mu`                      | Road friction coefficient.                                                |
| `C_Sf`                    | Front tire cornering stiffness.                                           |
| `C_Sr`                    | Rear tire cornering stiffness.                                            |
| `h`                       | Height of the center of gravity in meters.                                |
| `I`                       | Moment of inertia in kgm^2.                                               |
| `sv_min`                  | Minimum steering velocity in rad/s.                                       |
| `sv_max`                  | Maximum steering velocity in rad/s.                                       |
| `v_switch`                | Velocity switch parameter.                                                |
| `v_min`                   | Minimum velocity in m/s.                                                  |
| `v_max`                   | Maximum velocity in m/s.                                                  |
| `gym_rendering`           | Boolean flag to enable rendering of the Gym environment.                  |

## Inputs (Subscribed Topics)

The `GymBridge` node subscribes to the following ROS 2 topics:

| Topic Name                  | Message Type                            | Description                                        |
| :-------------------------- | :-------------------------------------- | :------------------------------------------------- |
| `/initialpose`              | `geometry_msgs/PoseWithCovarianceStamped` | Used to reset the ego vehicle's pose in the simulation. |
| `/control/command/control_cmd` | `autoware_auto_control_msgs/AckermannControlCommand` | Receives steering angle and speed commands for the ego vehicle. |
| `/cmd_vel`                  | `geometry_msgs/Twist`                   | (Optional) Used for keyboard teleoperation if `kb_teleop` is enabled. |

## Outputs (Published Topics)

The `GymBridge` node publishes to the following ROS 2 topics:

| Topic Name                | Message Type                | Description                                                          |
| :------------------------ | :-------------------------- | :------------------------------------------------------------------- |
| `<ego_scan_topic>` (e.g., `/scan`) | `sensor_msgs/LaserScan`     | Publishes laser scan data from the ego vehicle in the simulation.    |
| `/localization/kinematic_state` | `nav_msgs/Odometry`         | Publishes odometry information (pose and twist) of the ego vehicle.  |
| `/clock`                  | `rosgraph_msgs/Clock`       | Publishes the simulated time for ROS 2.                              |
| `/tf`                     | `tf2_msgs/TFMessage`        | Publishes the `map` to `base_link` and `base_link` to `laser` transforms. |
