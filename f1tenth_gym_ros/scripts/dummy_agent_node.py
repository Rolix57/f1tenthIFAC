#!/usr/bin/env python
import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import sensor_msgs.point_cloud2 as pc2
import laser_geometry.laser_geometry as lg
import csv
import math
from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import time

lp = lg.LaserProjection()

class Agent(object):

    def __init__(self):

        self.current_x = 0
        self.current_y = 0
        self.current_yaw = 0
        self.current_v = 0
        self.yaw_1 = 0
 
        self.drive_pub = rospy.Publisher('/drive', AckermannDriveStamped, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

        #with open('/home/rolando/catkin_ws/src/f1tenth_gym_ros/scripts/goals2.csv', 'r') as csvfile:
        with open('goals2.csv', 'r') as csvfile:
            self.goals = np.array(list(csv.reader(csvfile))).astype(np.float)

        self.k_1 = 0
        self.k = time.time()
        self.dt = 0

        self.x = np.array([self.current_x, self.current_y, self.current_yaw, 5.0, 0.0])
        self.goal = self.goals[0,:]

        self.config = Config()
        self.config.robot_type = RobotType.rectangle
        self.trajectory = np.array(self.x )
        self.recalculate = 0
        self.w = 0
        self.u = [5, 0]
        self.goal_idx = 0
        self.e_1 = 0

    def scan_callback(self, scan_msg):
        # print('got scan, now plan')
        pc2_msg = lp.projectLaser(scan_msg)
        point_generator = pc2.read_points(pc2_msg)

        point_cloud = list()
        i = 0
        for point in point_generator:
            if i % 2 == 0:
                point_cloud.append([point[0],point[1]])
            i += 1      

        ob = np.array(point_cloud)
        #ob = np.array([[]])

        dist_to_goal = np.hypot(self.current_x-self.goal[0],self.current_y-self.goal[1])
        if dist_to_goal < 2.0:
            self.goal_idx += 1
            self.goal = self.goals[self.goal_idx,:]
        #print(self.goal)

        if self.recalculate % 6 == 0:
            self.k_1 = self.k
            self.k = time.time()
            self.dt = self.k - self.k_1
            #self.config.dt = self.dt
 
            self.u, predicted_trajectory = dwa_control(self.x, self.config, self.goal, ob)
#            print(predicted_trajectory)
#            print()

            ## stanley
            waypoints = predicted_trajectory[:,:2]

            x_0, y_0 = waypoints[0,0], waypoints[0,1]
            try:
                x_1, y_1 = waypoints[1,0], waypoints[1,1]
            except:
                x_1, y_1 = waypoints[-1,0], waypoints[-1,1]

            a = (y_1 - y_0) / (x_1 - x_0)
            b = -1
            c = y_0 - a * x_0

            e = (a*self.current_x + b*self.current_y + c) / ((a**2 + b**2)**0.5)

            # heading error
            gamma = np.arctan2(waypoints[-1,1]-waypoints[0,1],
                               waypoints[-1,0]-waypoints[0,0])
            # h_e = np.arctan2(-a, b) - yaw

            h_e = gamma - self.current_yaw
            h_e = h_e - (2.0 * np.pi) if h_e > np.pi else h_e
            h_e = h_e + (2.0 * np.pi) if h_e < -np.pi else h_e

            # angulo entre coche y waypoint
            beta = np.arctan2(self.current_y - waypoints[0,1], self.current_x - waypoints[0,0])
            epsilon = gamma - beta
            epsilon = epsilon - (2.0 * np.pi) if epsilon > np.pi else epsilon
            epsilon = epsilon + (2.0 * np.pi) if epsilon < -np.pi else epsilon

            e = abs(e) if epsilon >= 0 else -abs(e)
            p_e = 0.3 * e
            i_e = 0.015 * (e + self.e_1) * self.dt
            d_e = 0.1 * (e - self.e_1) / self.dt

            theta_2 = np.arctan((p_e + i_e + d_e)/ (1 + self.current_v))
            # delta = -yaw + theta_2
            self.delta = h_e + theta_2
            self.delta = 1.22 if self.delta > 1.22 else self.delta
            self.delta = -1.22 if self.delta < -1.22 else self.delta

            self.e_1 = e

            ## imprimir

            #plt.cla()

            # for stopping simulation with the esc key.
            '''
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")
            plt.plot(self.x[0], self.x[1], "xr")
            plt.plot(self.goal[0], self.goal[1], "xb")
            plt.plot(ob[:, 0] + self.current_x, ob[:, 1] + self.current_y, "ok")

            
            plt.axis("equal")
            plt.draw()
            plt.grid(True)
            plt.pause(0.00000001)
            plt.clf()
            '''

        #print('delta', self.delta, np.degrees(self.delta))
        #vel = self.u[0] if self.u[0] <= 4.0 else 4.0
        #if abs(self.delta) > 1.1:
        #    self.u[0] = 1.5

        delta_yaw = self.current_yaw - self.yaw_1
        delta_yaw = delta_yaw - (2.0 * np.pi) if delta_yaw > np.pi else delta_yaw
        delta_yaw = delta_yaw + (2.0 * np.pi) if delta_yaw < -np.pi else delta_yaw
        self.w = delta_yaw / self.dt

        #print(self.u[0], self.delta)

        drive = AckermannDriveStamped()
        drive.drive.speed = self.u[0]
        drive.drive.steering_angle = self.delta
        self.drive_pub.publish(drive)

        #self.x = motion(self.x, u, self.dt)  # simulate robot
        self.x = np.array([self.current_x, self.current_y, self.current_yaw, self.current_v, self.w ])
        self.recalculate += 1

    def odom_callback(self, odom_msg):

        self.current_x = odom_msg.pose.pose.position.x
        self.current_y = odom_msg.pose.pose.position.y
        self.current_v_x = odom_msg.twist.twist.linear.x
        self.current_v_y = odom_msg.twist.twist.linear.y
        self.current_v = (self.current_v_x**2 + self.current_v_y**2) ** 0.5
        self.current_v_th = np.arctan2(self.current_v_y, self.current_v_x)
        #self.current_yaw = odom_msg.twist.twist.angular.z
        orientation_q = odom_msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        self.yaw_1 = self.current_yaw
        (self.current_roll, self.current_pitch, self.current_yaw) = euler_from_quaternion (orientation_list)

def dwa_control(x, config, goal, ob):
    """
    Dynamic Window Approach control
    """

    dw = calc_dynamic_window(x, config)
    u, trajectory = calc_control_and_trajectory(x, dw, config, goal, ob)
    return u, trajectory

def motion(x, u, dt):
    """
    motion model
    """
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt
    x[3] = u[0]
    x[4] = u[1]
    return x

def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    calculation final input with dynamic window
    """
    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):
            trajectory = predict_trajectory(x_init, v, y, config)
            # calc cost
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            final_cost = to_goal_cost + speed_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory
    return best_u, best_trajectory

def predict_trajectory(x_init, v, y, config):
    """
    predict trajectory with an input
    """
    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v, y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt
    return trajectory
 
def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """
    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
    return dw

def calc_to_goal_cost(trajectory, goal):
    """
        calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    error_angle = math.atan2(dy, dx)
    cost_angle = error_angle - trajectory[-1, 2]
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
    return cost

def calc_obstacle_cost(trajectory, ob, config):
    """
        calc obstacle cost inf: collision
    """
    ox = ob[:, 0]
    oy = ob[:, 1]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        #local_ob = np.array([local_ob @ x for x in rot])
        local_ob = np.array([local_ob.dot(x) for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2

        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")

    elif config.robot_type == RobotType.circle:
        if np.array(r <= config.robot_radius).any():
            return float("Inf")
    min_r = np.min(r)
    return 1.0 / min_r  # OK

class RobotType(Enum):
    circle = 0
    rectangle = 1

class Config:
    """
    simulation parameter class
    """
    def __init__(self):
        # robot parameter


        self.max_speed = 20/3.6  # [m/s] 20.0/3.6
        self.min_speed = 0.5  # [m/s]
        self.max_yaw_rate = 180.0 * math.pi / 180.0  # [rad/s] 170
        self.max_accel = 9.0  # [m/ss] 11,0
        self.max_delta_yaw_rate = 300.0 * math.pi / 180.0  # [rad/ss] 170
        self.v_resolution = 0.8  # [m/s] 0.8
        self.yaw_rate_resolution = 18.0 * math.pi / 180.0  # [rad/s] 30
        self.dt = 0.30  # [s] Time tick for motion prediction
        self.predict_time = 0.5  # [s]
        self.to_goal_cost_gain = 1.6 #1.4
        self.speed_cost_gain = 1.0 #1.0
        self.obstacle_cost_gain = 2.2 #2.0
        self.robot_type = RobotType.rectangle


        # if robot_type == RobotType.circle
        # Also used to check if goal is reached in both types
        #self.robot_radius = 1.00  # [m] for collision check

        # if robot_type == RobotType.rectangle
        self.robot_width = 1.1 # [m] for collision check
        self.robot_length = 1.1  # [m] for collision check

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value

if __name__ == '__main__':
    rospy.init_node('dummy_agent')
    dummy_agent = Agent()
    rospy.spin()
