# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from rclpy.time import Time
import rclpy
import rclpy.logging
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from tf2_ros import TransformBroadcaster
from rosgraph_msgs.msg import Clock
from autoware_auto_control_msgs.msg import AckermannControlCommand

import gym
import numpy as np
from transforms3d import euler

class GymBridge(Node):
    def __init__(self):
        super().__init__('gym_bridge')
        # default parameters
        self.declare_parameter('ego_scan_topic')
        self.declare_parameter('scan_distance_to_base_link')
        self.declare_parameter('scan_fov')
        self.declare_parameter('scan_beams')
        self.declare_parameter('map_path')
        self.declare_parameter('map_img_ext')
        self.declare_parameter('sx')
        self.declare_parameter('sy')
        self.declare_parameter('stheta')
        self.declare_parameter('kb_teleop')
        # add parameters
        # custom parameters
        self.declare_parameter("mass")
        self.declare_parameter("length")
        self.declare_parameter("width")
        self.declare_parameter("lf")
        self.declare_parameter("lr")
        self.declare_parameter("s_max")
        self.declare_parameter("s_min")
        self.declare_parameter("a_max")
        self.declare_parameter("mu")
        self.declare_parameter("C_Sf")
        self.declare_parameter("C_Sr")
        self.declare_parameter("h")
        self.declare_parameter("I")
        self.declare_parameter("sv_min")
        self.declare_parameter("sv_max")
        self.declare_parameter("v_switch")
        self.declare_parameter("v_min")
        self.declare_parameter("v_max")

        # rendering parameters
        self.declare_parameter('gym_rendering')

        # get parameters
        self.m = self.get_parameter('mass').value
        self.length = self.get_parameter('length').value
        self.width = self.get_parameter('width').value
        self.lf = self.get_parameter('lf').value
        self.lr = self.get_parameter('lr').value
        self.s_max = self.get_parameter('s_max').value
        self.s_min = self.get_parameter('s_min').value
        self.a_max = self.get_parameter('a_max').value
        self.mu = self.get_parameter('mu').value
        self.C_Sf = self.get_parameter('C_Sf').value
        self.C_Sr = self.get_parameter('C_Sr').value
        self.h = self.get_parameter('h').value
        self.I = self.get_parameter('I').value
        self.sv_min = self.get_parameter('sv_min').value
        self.sv_max = self.get_parameter('sv_max').value
        self.v_switch = self.get_parameter('v_switch').value
        self.v_min = self.get_parameter('v_min').value
        self.v_max = self.get_parameter('v_max').value

        self.rendering = self.get_parameter('gym_rendering').value


        # custom_params = {
        #     # ユーザー指定のパラメータ
        #     'm': 160.0,                               # 質量 [kg]
        #     'length': 2.0,                            # 全長 [m]
        #     'width': 1.45,                            # 全幅 [m]
        #     'lf': 1.087/2.0,                        # 重心から前軸までの距離 [m] (ホイールベースの半分と仮定)
        #     'lr': 1.087/2.0,                        # 重心から後軸までの距離 [m] (ホイールベースの半分と仮定)
        #     's_max': 0.64,         # 最大ステアリング角 [rad]
        #     's_min': -0.64,        # 最小ステアリング角 [rad]
        #     'a_max': 3.2,                             # 最大加速度 [m/s^2]
            
        #     # デフォルトまたは推定値を使用するパラメータ
        #     'mu': 1.0489,                             # 路面摩擦係数 (デフォルト値)
        #     'C_Sf': 5.64718,                            # 前輪コーナリングスティフネス (デフォルト値)
        #     'C_Sr': 5.65456,                           # 後輪コーナリングスティフネス (デフォルト値)
        #     'h': 0.2,                                 # 重心の高さ [m] (推定値)
        #     'I': 81.37,           # 慣性モーメント [kgm^2] (均一な棒として概算)
        #     'sv_min': -0.32,                          # 最小ステアリング速度 [rad/s] (デフォルト値)
        #     'sv_max': 0.32,                            # 最大ステアリング速度 [rad/s] (デフォルト値)
        #     'v_switch': 7.319,                        # 速度スイッチ (デフォルト値)
        #     'v_min': -10.0,                           # 最小速度 [m/s] (調整値)
        #     'v_max': 25.0,                            # 最大速度 [m/s] (調整値)
        # }
        custom_params = {
            'm': self.m,                               # 質量 [kg]
            'length': self.length,                     # 全長 [m]
            'width': self.width,                       # 全幅 [m]
            'lf': self.lf,                             # 重心から前軸までの距離 [m]
            'lr': self.lr,                             # 重心から後軸までの距離 [m]
            's_max': self.s_max,                       # 最大ステアリング角 [rad]
            's_min': self.s_min,                       # 最小ステアリング角 [rad]
            'a_max': self.a_max,                       # 最大加速度 [m/s^2]
            'mu': self.mu,                             # 路面摩擦係数
            'C_Sf': self.C_Sf,                         # 前輪コーナリングスティフネス
            'C_Sr': self.C_Sr,                        # 後輪コーナリングスティフネス
            'h': self.h,                               # 重心の高さ [m]
            'I': self.I,                               # 慣性モーメント [kgm^2]
            'sv_min': self.sv_min,                     # 最小ステアリング速度 [rad/s]
            'sv_max': self.sv_max,                     # 最大ステアリング速度 [rad/s]
            'v_switch': self.v_switch,                 # 速度スイッチ
            'v_min': self.v_min,                       # 最小速度 [m/s]
            'v_max': self.v_max,                       # 最大速度 [m/s]
        }

        # env backend
        self.env = gym.make('aic_gym:aic-v0',
                            map=self.get_parameter('map_path').value,
                            map_ext=self.get_parameter('map_img_ext').value,
                            num_agents=1, params=custom_params)
        def render_callback(env_renderer):
            # custom extra drawing function

            e = env_renderer

            # update camera to follow car
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)

            e.left = left - 1200
            e.right = right + 1200
            e.top = top + 1200
            e.bottom = bottom - 1200
        
        self.env.add_render_callback(render_callback)

        sx = self.get_parameter('sx').value
        sy = self.get_parameter('sy').value
        stheta = self.get_parameter('stheta').value
        self.ego_pose = [sx, sy, stheta]
        self.ego_speed = [0.0, 0.0, 0.0]
        self.ego_requested_speed = 0.0
        self.ego_steer = 0.0
        self.ego_collision = False
        ego_scan_topic = self.get_parameter('ego_scan_topic').value
        scan_fov = self.get_parameter('scan_fov').value
        scan_beams = self.get_parameter('scan_beams').value
        self.angle_min = -scan_fov / 2.
        self.angle_max = scan_fov / 2.
        self.angle_inc = scan_fov / scan_beams
        ego_odom_topic = "/localization/kinematic_state"
        self.scan_distance_to_base_link = self.get_parameter('scan_distance_to_base_link').value
        self.get_logger().info(f"{sx}, {sy}, {stheta} initialize")
        self.obs, _ , self.done, _ = self.env.reset(np.array([[sx, sy, stheta]]))
        self.ego_scan = list(self.obs['scans'][0])

        # sim physical step timer
        self.drive_timer = self.create_timer(0.01, self.drive_timer_callback)
        # topic publishing timer
        self.timer = self.create_timer(0.004, self.timer_callback)

        # transform broadcaster
        self.br = TransformBroadcaster(self)

        # publishers
        self.ego_scan_pub = self.create_publisher(LaserScan, ego_scan_topic, 10)
        self.ego_odom_pub = self.create_publisher(Odometry, ego_odom_topic, 10)
        self.publisher_ = self.create_publisher(Clock, '/clock', 10)
        self.ego_drive_published = False
        self.reset_pose = [sx, sy, stheta]
        self.start_time = self.get_clock().now().nanoseconds / 1e9
        self.ts = Time(seconds=0, nanoseconds=0)

        # subscribers
        self.ego_reset_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.ego_reset_callback,
            10)

        self.subscription = self.create_subscription(
            AckermannControlCommand,
            '/control/command/control_cmd',
            self.control_command_callback,
            10 # QoS history depth
        )

        if self.get_parameter('kb_teleop').value:
            self.teleop_sub = self.create_subscription(
                Twist,
                '/cmd_vel',
                self.teleop_callback,
                10)

    def control_command_callback(self, msg: AckermannControlCommand):
        """
        Callback function for the /control/command/control_cmd topic.
        Extracts steering angle and speed to control the gym environment.
        """
        self.ego_steer = msg.lateral.steering_tire_angle
        self.ego_requested_speed = msg.longitudinal.speed
        self.ego_drive_published = True



    def opp_drive_callback(self, drive_msg):
        self.opp_requested_speed = drive_msg.drive.speed
        self.opp_steer = drive_msg.drive.steering_angle
        self.opp_drive_published = True

    def ego_reset_callback(self, pose_msg):
        rx = pose_msg.pose.pose.position.x
        ry = pose_msg.pose.pose.position.y
        rqx = pose_msg.pose.pose.orientation.x
        rqy = pose_msg.pose.pose.orientation.y
        rqz = pose_msg.pose.pose.orientation.z
        rqw = pose_msg.pose.pose.orientation.w
        _, _, rtheta = euler.quat2euler([rqw, rqx, rqy, rqz], axes='sxyz')
        self.obs, _ , self.done, _ = self.env.reset(np.array([[rx, ry, rtheta]]))
        self.reset_pose = [rx, ry, rtheta]

    def teleop_callback(self, twist_msg):
        if not self.ego_drive_published:
            self.ego_drive_published = True

        self.ego_requested_speed = twist_msg.linear.x

        if twist_msg.angular.z > 0.0:
            self.ego_steer = 0.3
        elif twist_msg.angular.z < 0.0:
            self.ego_steer = -0.3
        else:
            self.ego_steer = 0.0

    def drive_timer_callback(self):
        if self.ego_drive_published:
            self.obs, _, self.done, _ = self.env.step(np.array([[self.ego_steer, self.ego_requested_speed]]))
        if self.rendering:
            self.env.render(mode='human')
        self._update_sim_state()
        # publish clock
        msg = Clock()
        current_relative_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time
        sec = int(current_relative_time)
        nanosec = int((current_relative_time - sec) * 1e9)
        msg.clock.sec = sec
        msg.clock.nanosec = nanosec
        self.ts = Time(seconds=sec, nanoseconds=nanosec)
        self.publisher_.publish(msg)

    def timer_callback(self):

        # pub scans
        scan = LaserScan()
        scan.header.stamp = self.ts.to_msg()
        scan.header.frame_id = 'laser'
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_inc
        scan.range_min = 0.
        scan.range_max = 30.
        scan.ranges = self.ego_scan
        self.ego_scan_pub.publish(scan)
        ts = self.ts

        # pub tf
        self._publish_odom(ts)
        self._publish_transforms(ts)
        self._publish_laser_transforms(ts)

        if self.done:
            self.get_logger().info('Collision detected')
            # wait for 2 seconds
            time.sleep(2)
            self.done = False
            self.obs, _ , self.done, _ = self.env.reset(np.array([self.reset_pose]))

    def _update_sim_state(self):
        self.ego_scan = list(self.obs['scans'][0])

        self.ego_pose[0] = self.obs['poses_x'][0]
        self.ego_pose[1] = self.obs['poses_y'][0]
        self.ego_pose[2] = self.obs['poses_theta'][0]
        self.ego_speed[0] = self.obs['linear_vels_x'][0]
        self.ego_speed[1] = self.obs['linear_vels_y'][0]
        self.ego_speed[2] = self.obs['ang_vels_z'][0]

        

    def _publish_odom(self, ts):
        ego_odom = Odometry()
        ego_odom.header.stamp = ts.to_msg()
        ego_odom.header.frame_id = 'map'
        ego_odom.child_frame_id = 'base_link'
        ego_odom.pose.pose.position.x = self.ego_pose[0]
        ego_odom.pose.pose.position.y = self.ego_pose[1]
        ego_quat = euler.euler2quat(0., 0., self.ego_pose[2], axes='sxyz')
        ego_odom.pose.pose.orientation.x = ego_quat[1]
        ego_odom.pose.pose.orientation.y = ego_quat[2]
        ego_odom.pose.pose.orientation.z = ego_quat[3]
        ego_odom.pose.pose.orientation.w = ego_quat[0]
        ego_odom.twist.twist.linear.x = self.ego_speed[0]
        ego_odom.twist.twist.linear.y = self.ego_speed[1]
        ego_odom.twist.twist.angular.z = self.ego_speed[2]
        self.ego_odom_pub.publish(ego_odom)


    def _publish_transforms(self, ts):
        ego_t = Transform()
        ego_t.translation.x = self.ego_pose[0]
        ego_t.translation.y = self.ego_pose[1]
        ego_t.translation.z = 0.0
        ego_quat = euler.euler2quat(0.0, 0.0, self.ego_pose[2], axes='sxyz')
        ego_t.rotation.x = ego_quat[1]
        ego_t.rotation.y = ego_quat[2]
        ego_t.rotation.z = ego_quat[3]
        ego_t.rotation.w = ego_quat[0]

        ego_ts = TransformStamped()
        ego_ts.transform = ego_t
        ego_ts.header.stamp = ts.to_msg()
        ego_ts.header.frame_id = 'map'
        ego_ts.child_frame_id = 'base_link'
        self.br.sendTransform(ego_ts)


    def _publish_laser_transforms(self, ts):
        ego_scan_ts = TransformStamped()
        ego_scan_ts.transform.translation.x = self.scan_distance_to_base_link
        # ego_scan_ts.transform.translation.z = 0.04+0.1+0.025
        ego_scan_ts.transform.rotation.w = 1.
        ego_scan_ts.header.stamp = ts.to_msg()
        ego_scan_ts.header.frame_id = 'base_link'
        ego_scan_ts.child_frame_id = 'laser'
        self.br.sendTransform(ego_scan_ts)

def main(args=None):
    rclpy.init(args=args)
    gym_bridge = GymBridge()
    rclpy.spin(gym_bridge)

if __name__ == '__main__':
    main()
