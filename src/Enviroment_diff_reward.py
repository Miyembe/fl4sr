#! /usr/bin/env python3.8
import sys
import os
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
from math import sqrt
import time
import numpy as np
import rospy

from worlds import World
from InfoGetter import InfoGetter

from geometry_msgs.msg import Twist
from rospy.service import ServiceException
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.srv import StepControl
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion
import queue
from collections import deque

class Enviroment():
    """Similar class as openAI gym Env. 
    
    Is able to: reset, step.
    """

    def __init__(self, 
        world: World,
        reward_goal: float,
        reward_collision: float,
        reward_progress: float,
        reward_max_collision: float,
        list_reward: int,
        factor_linear: float,
        factor_angular: float,
        is_progress: bool = True
        ) -> None:
        """Initializes eviroment.

        Args:
            world (World): holds enviroment variables
        """
        print(f"INSIDE Enviroment | world: {world}, reward_goal: {reward_goal}, reward_collision: {reward_collision}, reward_progress: {reward_progress}, factor_linear: {factor_linear}, factor_angular: {factor_angular}, is_progress: {is_progress}")
        # params        
        self.COLLISION_RANGE = 0.20
        self.MAXIMUM_SCAN_RANGE = 1.0
        self.MINIMUM_SCAN_RANGE = 0.12
        self.GOAL_RANGE = 0.5
        #print(f"list_reward: {list_reward}")
        if list_reward == 1:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([-20, -20, -20, -20])
            self.ARR_REWARD_PROGRESS = np.array([40, 40, 40, 40])
        if list_reward == 2:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, -20, -50, -30])
            self.ARR_REWARD_PROGRESS = np.array([60, 40, 20, 40])
        if list_reward == 3:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, -30, -40, -60])
            self.ARR_REWARD_PROGRESS = np.array([60, 30, 40, 40])
        if list_reward == 4:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, -20, -30, -40])
            self.ARR_REWARD_PROGRESS = np.array([60, 10, 20, 20])
        if list_reward == 5:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, -30, -40, -60])
            self.ARR_REWARD_PROGRESS = np.array([60, 10, 20, 20])
        if list_reward == 6:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, -60, -40, -30])
            self.ARR_REWARD_PROGRESS = np.array([60, 30, 40, 40])
        if list_reward == 7:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, -60, -40, -30])
            self.ARR_REWARD_PROGRESS = np.array([60, 10, 20, 20])
        if list_reward == 8:
            self.ARR_REWARD_GOAL = np.array([100, 100, 100, 100])
            self.ARR_REWARD_COLLISION = np.array([0, 0, -20, -20])
            self.ARR_REWARD_PROGRESS = np.array([60, 0, 10, 10])
        self.REWARD_GOAL = reward_goal
        self.REWARD_COLLISION_FIX_RATE = 0.25
        self.REWARD_COLLISION_FIX = self.ARR_REWARD_COLLISION * self.REWARD_COLLISION_FIX_RATE
        self.REWARD_COLLISION_VARIABLE = self.ARR_REWARD_COLLISION * (1 - self.REWARD_COLLISION_FIX_RATE)
        self.REWARD_COLLISION = np.add(self.REWARD_COLLISION_FIX, self.REWARD_COLLISION_VARIABLE)
        self.REWARD_TIME = -0.3
        self.PROGRESS_REWARD_FACTOR = self.ARR_REWARD_PROGRESS
        self.FACTOR_LINEAR = factor_linear
        self.FACTOR_ANGULAR = 1.0 #factor_angular
        self.FACTOR_NORMALISE_DISTANCE = 5.0
        self.FACTOR_NORMALISE_ANGLE = np.pi
        self.REWARD_MAX_COLLISION_DENSE = reward_max_collision
        self.LAMBDA_COLLISION = np.log(self.REWARD_MAX_COLLISION_DENSE + 1)
        self.MAX_DISTANCE = 5.5
        self.IS_PROGRESS = is_progress
        if self.IS_PROGRESS == True:
            self.LAMBDA = np.log(2)/5 # np.log(5)/5 let's choose one
        else:
            self.LAMBDA = np.log(5)/5

        # simulation services
        # rospy.wait_for_service('/gazebo/reset_simulation')
        # self.reset_simulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        rospy.wait_for_service('/gazebo/pause_physics')
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.step_control = rospy.ServiceProxy('/gazebo/step_control', StepControl)
        self.sim_step_size = 0.01 
        self.pause()
        # world settings
        self.robot_count = world.robot_count
        self.robot_alives = world.robot_alives
        self.robot_indexes = world.robot_indexes

        self.x_starts_all = world.x_starts
        self.y_starts_all = world.y_starts

        self.x_starts = [x[0] if isinstance(x, float) is not True else x for x in world.x_starts]
        self.y_starts = [y[0] if isinstance(y, float) is not True else y for y in world.y_starts]

        self.targets = world.target_positions
        self.x_targets = np.array(self.targets).T[0]
        self.y_targets = np.array(self.targets).T[1]

        self.coordinates_arena = [[(0.0,5.0),(4.2,9.3)], [(0.0, 0.0), (4.2, 4.3)], [(-5.0,-5.0),(-0.8,-0.7)], [(-5.0,-10.0),(-0.8, -5.7)]]
        
        
        # self.start_indexes = [0 for _ in range(len(self.robot_count))]
        # self.target_indexes = [0 for _ in range(len(self.robot_count))]
        # create restart enviroment messages
        self.reset_tb3_messages = \
            [self.create_model_state('tb3_{}'.format(rid), 
                                     self.x_starts[id], 
                                     self.y_starts[id],
                                     -0.2)
            for id, rid in enumerate(self.robot_indexes)]
        self.reset_target_messages = \
            [self.create_model_state('target_{}'.format(rid), 
                                     self.targets[id][0], 
                                     self.targets[id][1], 
                                     0)
            for id, rid in enumerate(self.robot_indexes)]
        self.command_empty = Twist()
        # basic settings
        self.node = rospy.init_node('turtlebot_env', anonymous=True)
        self.rate_freq = 100
        self.rate_period = 1 / self.rate_freq
        self.rate = rospy.Rate(self.rate_freq)
        self.laser_count = 24
        
        self.observation_dimension = self.laser_count + 1 # 20230315 removed abs_distance
        self.action_dimension = 2

        # publishers for turtlebots
        self.publisher_turtlebots = \
            [rospy.Publisher('/tb3_{}/cmd_vel'.format(i), 
                             Twist, 
                             queue_size=1) 
            for i in self.robot_indexes]
        # positional info getter
        self.position_info_getter = InfoGetter()
        self._position_subscriber = rospy.Subscriber("/gazebo/model_states", 
                                                     ModelStates, 
                                                     self.position_info_getter)
        # lasers info getters, subscribers unused
        self.laser_info_getter = [InfoGetter() for i in range(self.robot_count)]
        self._laser_subscriber = \
            [rospy.Subscriber('/tb3_{}/scan'.format(rid), 
                              LaserScan, 
                              self.laser_info_getter[id]) 
            for id, rid in enumerate(self.robot_indexes)]
        
        self.num_laser_buffer = 10
        self.laser_buffer = [queue.Queue(self.num_laser_buffer) for i in range(self.robot_count)] #[[] for i in range(self.robot_count)]

        # various simulation outcomes
        self.robot_finished = np.zeros((self.robot_count), dtype=bool)
        self.robot_succeeded = np.zeros((self.robot_count), dtype=bool)
        # previous and current distances
        self.robot_target_distances_previous = self.get_distance(
            self.x_starts, 
            self.x_targets, 
            self.y_starts, 
            self.y_targets)
        
        return

    def check_outside_arena(self,
        x: list,
        y: list,
        idx: int
        ) -> bool:
        
        coordinates = self.coordinates_arena[idx]
        if (x[idx] > coordinates[0][0] and x[idx] < coordinates[1][0]) and \
           (y[idx] > coordinates[0][1] and y[idx] < coordinates[1][1]):
            return False
        else: return True

    def reset(self,
        robot_id: int=-1
        ) -> None:
        """Resets robots to starting state.
        If robot_id is empty all robots will be reseted.

        Args:
            robot_id (int, optional): Id of robot to reset. Defaults to -1.
        """

        # wait for services
        rospy.wait_for_service('/gazebo/reset_simulation')
        rospy.wait_for_service('/gazebo/set_model_state')

        # Reset laser_buffer (for median filter)
        self.laser_buffer = [queue.Queue(self.num_laser_buffer) for i in range(self.robot_count)] #[[] for i in range(self.robot_count)]
        # set model states or reset world
        if robot_id == -1:
            try:
                state_setter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                for id, rid in enumerate(self.robot_indexes):
                    # pick new starting position and direction and set them
                    # TODO several starting and target points. 
                    # if self.start_indexes[id] < len(self.start_indexes[id]):
                    #     self.start_indexes[id] = self.start_indexes[id] + 1
                    # else: self.start_indexes[id] = 0
                    # if self.target_indexes[id] < len(self.target_indexes[id]):
                    #     self.target_indexes[id] = self.target_indexes[id] + 1
                    # else: self.target_indexes[id] = 0    
                        
                    start_index = np.random.randint(len(self.x_starts_all[id]))
                    #target_index = np.random.randint(len(self.target_positions[id]))
                    self.x_starts[id] = self.x_starts_all[id][start_index]
                    self.y_starts[id] = self.y_starts_all[id][start_index]
                    direction = 0.0 #+ (np.random.rand() * np.pi / 2) - (np.pi / 4)
                    # generate new message
                    self.reset_tb3_messages[id] = \
                        self.create_model_state('tb3_{}'.format(rid), 
                                             self.x_starts[id], 
                                             self.y_starts[id],
                                             direction)
                    # reset enviroment position
                    state_setter(self.reset_tb3_messages[id])
                    #state_setter(self.reset_target_messages[id])
                    self.robot_finished[id] = False
                print('Starts x:', self.x_starts)
                print('Starts y:', self.y_starts)
            except rospy.ServiceException as e:
                print('Failed state setter!', e)
            #self.reset_world()
        else:
            try:
                state_setter = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
                state_setter(self.reset_tb3_messages[robot_id])
                #state_setter(self.reset_target_messages[robot_id])
                self.robot_finished[robot_id] = False
            except rospy.ServiceException as e:
                print('Failed state setter!', e)
        # set robot move command
        if robot_id == -1:
            for i in range(self.robot_count):
                self.publisher_turtlebots[i].publish(self.command_empty)
                self.robot_target_distances_previous = self.get_distance(
                    self.x_starts, 
                    self.x_targets, 
                    self.y_starts, 
                    self.y_targets)
                self.robot_succeeded = np.zeros((self.robot_count), dtype=bool)
        else:
            self.publisher_turtlebots[robot_id].publish(self.command_empty)            
            self.robot_target_distances_previous[robot_id] = \
                sqrt(
                     (self.x_starts[robot_id] - self.x_targets[robot_id])**2 
                     + (self.y_starts[robot_id] - self.y_targets[robot_id])**2)
        # wait for new scan message, so that laser values are updated
        # kinda cheeky but it works on my machine :D
        #print(f"before unpause")
        self.unpause()
        #print(f"before scan")
        
        rospy.wait_for_message('/tb3_{}/scan'.format(self.robot_indexes[0]), LaserScan)
        #print(f"before pause")
        self.pause()
        #print(f"end of reset")
        return
        
    
    def step(self,
        actions: np.ndarray,
        time_step: float=0.3
        ) -> tuple:
        """Perform one step of simulations using given actions and lasting for 
        set time.

        Args:
            actions (np.ndarray): Action of each robot.
            time_step (float, optional): Duration of taken step. Defaults to 0.1.

        Returns:
            tuple: states (np.ndarray), 
                   rewards (np.ndarray), 
                   robot_finished (list), 
                   robot_succeeded (list),
                   error (bool)
                   data (dict)
        """
        assert len(actions) == self.robot_count, 'Wrong actions dimension!'
        # generate twists, also get separate values of actions
        twists = [self.action_to_twist(action) for action in actions]
        #forward_vel = Twist()
        #forward_vel.linear.x = 0.25
        #forward_vel.angular.z = 0.0
        #twists = [forward_vel for i in range(self.robot_count)]
        actions_linear_x = actions.T[1]
        actions_angular_z = actions.T[0]
        # publish twists
        for i in range(self.robot_count):
            self.publisher_turtlebots[i].publish(twists[i])
            #self.publisher_turtlebots[i].publish(Twist())
        # start of timing !!! changed to rospy time !!!
        
        running_time = 0
        # move robots with action for time_step        
        
        model_state_before_time_step = self.position_info_getter.get_msg()
        robot_indexes_b = self.get_robot_indexes_from_model_state(model_state_before_time_step)
        x_b, y_b, _, _ = self.get_positions_from_model_state(model_state_before_time_step, robot_indexes_b)
        start_time = rospy.get_time()
        start_time_real = time.time()
        #self.unpause()
        num_simulation_step = int(time_step / self.sim_step_size)
        elapsed_sim_time = 0
        self.step_control(True, True, num_simulation_step)
        
        while elapsed_sim_time < time_step - self.rate_period:
            self.rate.sleep() 
            elapsed_sim_time = rospy.get_time() - start_time

        # self.unpause()
        # while_loop_counter = 0
        # while(running_time < time_step):
            
        #     self.rate.sleep()
        #     running_time = rospy.get_time() - start_time
        #     running_time_real = time.time() - start_time_real
        #     while_loop_counter = while_loop_counter + 1
        #     #print(f"Running Time (rospy.get_time()): {running_time} ")
        #     #print(f"Running Time (time.time()): {running_time_real}")
        #     #print(f"While loop counter: {while_loop_counter}")
        
        # self.pause()
        

        model_state_after_time_step = self.position_info_getter.get_msg()
        robot_indexes_a = self.get_robot_indexes_from_model_state(model_state_after_time_step)
        x_a, y_a, _, _ = self.get_positions_from_model_state(model_state_after_time_step, robot_indexes_a)
        robot_dist_time_step = self.get_distance(x_a, x_b,
                                                 y_a, y_b)
        #print(f"Traversed Distances for each time step: {robot_dist_time_step}")
        # send empty commands to robots
        # self.unpause()
        # read current positions of robots
        model_state = self.position_info_getter.get_msg()
        robot_indexes = self.get_robot_indexes_from_model_state(model_state)
        x, y, theta, correct = self.get_positions_from_model_state(model_state, 
                                                                   robot_indexes)
        # check for damaged robots
        if np.any(np.isnan(correct)):
            print('ERROR: Enviroment: nan robot twist detected!')
            return None, None, None, None, True, None

        theta = theta % (2 * np.pi)
        # get current distance to goal
        robot_target_distances = self.get_distance(x, self.x_targets, 
                                                   y, self.y_targets)
        # get current robot angles to targets
        robot_target_angle = self.get_angle(self.x_targets, x, 
                                            self.y_targets, y)
        robot_target_angle = robot_target_angle % (2 * np.pi)
        robot_target_angle_difference = (robot_target_angle - theta - np.pi) % (2 * np.pi) - np.pi
        # get current laser measurements
        # for _ in range(self.num_laser_buffer):
        #     scan = self.laser_info_getter[i].get_msg()
        #     for i in range(self.robot_count):
        #         self.laser_buffer[i].append(scan.ranges[i])

        self.put_scan_into_buffer()
        list_median_scan_ranges = self.apply_median_filter()
        robot_lasers, robot_collisions, id_collisions = self.get_robot_lasers_collisions(list_median_scan_ranges)

        # create state array 
        # = lasers (24), 
        #   action linear x (1), action angular z (1), 
        #   distance to target (1), angle to target (1)
        # = dimension (6, 28)
        s_actions_linear = actions_linear_x.reshape((self.robot_count, 1))
        s_actions_angular = actions_angular_z.reshape((self.robot_count, 1))
        s_robot_target_distances = robot_target_distances.reshape((self.robot_count, 1)) /self.FACTOR_NORMALISE_DISTANCE 
        s_robot_target_angle_difference = robot_target_angle_difference.reshape((self.robot_count, 1)) / self.FACTOR_NORMALISE_ANGLE
        assert robot_lasers.shape == (self.robot_count, 24), f'Wrong lasers dimension!: {robot_lasers.shape}'
        assert s_actions_linear.shape == (self.robot_count, 1), 'Wrong action linear dimension!'
        assert s_actions_angular.shape == (self.robot_count, 1), 'Wrong action angular dimension!'
        assert s_robot_target_distances.shape == (self.robot_count, 1), 'Wrong distance to target!'
        assert s_robot_target_angle_difference.shape == (self.robot_count, 1), 'Wrong angle to target!'
        states = np.hstack((robot_lasers, 
                            #s_actions_linear, s_actions_angular, 
                            #s_robot_target_distances, 
                            s_robot_target_angle_difference))
        assert states.shape == (self.robot_count, self.observation_dimension), 'Wrong states dimension!'
        
        # rewards
        # distance rewards
        # CHECK for possible huge value after reset

        # if self.IS_PROGRESS == True:
        #     exp_progress_factor = (2-np.exp(self.LAMBDA * self.MAX_DISTANCE))/2 
        #     reward_distance = exp_progress_factor * self.PROGRESS_REWARD_FACTOR * (self.robot_target_distances_previous - robot_target_distances)
        # else: 
        #     reward_distance = 1 - np.exp(self.LAMBDA * self.MAX_DISTANCE)
        reward_distance = np.zeros(self.robot_count)
        progress_distance = np.zeros(self.robot_count)
        for i in range(self.robot_count):
            progress_distance[i] = self.robot_target_distances_previous[i] - robot_target_distances[i]
            if progress_distance[i] > 0:
                reward_distance[i] = self.PROGRESS_REWARD_FACTOR[i] * (progress_distance[i])
            else:
                reward_distance[i] = self.PROGRESS_REWARD_FACTOR[i] * (progress_distance[i])#reward_distance[i] = 0
        
        # reward_distance = - np.e ** (0.25 * robot_target_distances)
        # goal reward
        reward_goal = np.zeros(self.robot_count)
        for i in range(self.robot_count):
            if robot_target_distances[i] < self.GOAL_RANGE:
                reward_goal[i] = self.ARR_REWARD_GOAL[i]
            #reward_goal[robot_target_distances < self.GOAL_RANGE] = self.REWARD_GOAL
        self.robot_finished[robot_target_distances < self.GOAL_RANGE] = True
        self.robot_succeeded[robot_target_distances < self.GOAL_RANGE] = True
        # collision reward
        reward_collision = self.calculate_reward_collision(robot_collisions, id_collisions, robot_lasers)
        #reward_collision[np.where(robot_collisions)] = self.REWARD_COLLISION
        reward_time = self.REWARD_TIME
        self.robot_finished[np.where(robot_collisions)] = True
        # total reward
        rewards = reward_distance + reward_goal #+ #reward_collision + reward_time
        #print(f"robot_collisions: {robot_collisions}")
        #print(f"robot_lasers: {robot_lasers}")
        print(f"rewards: {rewards}")
        print(f"reward_distance: {reward_distance}")
        print(f"progress_distance: {progress_distance}")
        print(f"reward_collision: {reward_collision}")
        #print(f"reward_time: {reward_time}")
        print(f"reward_goal: {reward_goal}")
        
        # set current target distance as previous
        distances_help = self.robot_target_distances_previous.copy()
        self.robot_target_distances_previous = robot_target_distances.copy()
        # restart robots
        was_restarted = False
        robot_finished = self.robot_finished.copy()
        for i in range(self.robot_count):
            # I will add the robot out of the arena here
            check_outside = self.check_outside_arena(x, y, i)
            if self.robot_finished[i] or check_outside:
                self.reset(i)
                was_restarted = True
        if was_restarted:
            states = self.get_current_states()
        # additional data to send
        data = {}
        data['x'] = x
        data['y'] = y
        data['theta'] = theta

        return states, rewards, robot_finished, self.robot_succeeded, False, data

    def calculate_reward_collision(self, robot_collisions, id_collisions, robot_lasers):
        # 1. Reward for the collision event
        reward_collision = np.zeros(self.robot_count)
        for idx in np.where(robot_collisions)[0]:
            reward_collision[idx] = (self.REWARD_COLLISION_VARIABLE[idx] * abs(np.cos(id_collisions[idx]/self.laser_count)) - self.REWARD_COLLISION_FIX[idx])
        # 2. Reward for the dense collision reward - suppresses more progressive reward
        for jdx in range(self.robot_count):
            reward_collision[jdx] += -(np.exp(robot_lasers[jdx][id_collisions[jdx]] * self.LAMBDA_COLLISION)) + 1
        
        return reward_collision
        
        # 
    def create_model_state(self, 
        name: str,
        pose_x: float,
        pose_y: float,
        orientation_z: float,
        ) -> ModelState:
        """Creates basic ModelState with specified values. 
        Other values are set to zero.

        Args:
            name (str): Name.
            pose_x (float): Value of ModelState.pose.x
            pose_y (float): Value of ModelState.pose.y
            orientation_z (float): Value of ModelState.oritentation.z

        Returns:
            ModelState: Initialized model state.
        """
        model_state = ModelState()
        model_state.model_name = name
        model_state.pose.position.x = pose_x
        model_state.pose.position.y = pose_y
        model_state.pose.position.z = 0
        model_state.pose.orientation.x = 0.0
        model_state.pose.orientation.y = 0.0
        model_state.pose.orientation.z = orientation_z
        model_state.pose.orientation.w = 0.0
        return model_state

    def get_robot_indexes_from_model_state(self,
        model_state: ModelStates=None
        ) -> list:
        """Creates list with indexes of robots in model state.

        Args:
            model_state (ModelStates, optional): Source of robot indexes. Defaults to None.

        Returns:
            list: Robot indexes. ('tb_2' index is list[2])
        """
        robots = [None for i in range(len(self.robot_alives))]
        if model_state is None:
            model_state = self.position_info_getter.get_msg()
        for i in range(len(model_state.name)):
            if 'tb3' in model_state.name[i]:

                robots[int(model_state.name[i][-1])] = i
        return robots

    def action_to_twist(self,
        action: np.ndarray,
        ) -> Twist:
        """Transforms action 2d ndarray to Twist message.

        Args:
            action (np.ndarray): Ndarray 2d.

        Returns:
            Twist: Transformed message from ndarray.
        """
        assert len(action) == 2, 'Wrong action dimension!'
        twist = Twist()
        twist.linear.x = action[1] * self.FACTOR_LINEAR
        twist.angular.z = action[0] * self.FACTOR_ANGULAR
        return twist

    def get_distance(self, 
        x_0: np.ndarray,
        x_1: np.ndarray,
        y_0: np.ndarray,
        y_1: np.ndarray
        ) -> np.ndarray:
        """Returns distance between two arrays of positions.

        Args:
            x_0 (np.ndarray): X positions of first points
            x_1 (np.ndarray): X positions of second points
            y_0 (np.ndarray): Y positions of first points
            y_1 (np.ndarray): Y positions of second points

        Returns:
            np.ndarray: Distances between points.
        """
        return np.sqrt((np.square(x_0 - x_1) + np.square(y_0 - y_1))) 

    def get_positions_from_model_state(self,
        model_state: ModelStates,
        robot_indexes: list
        ) -> tuple:
        """Get positional information from model_state

        Args:
            model_state (ModelStates): Information source.
            robot_indexes (list): List of robot indexes.

        Returns:
            tuple: x, y, theta ndarrays of robots
        """
        x, y, theta, correct = [], [], [], []
        for rid in self.robot_indexes:
            index = robot_indexes[rid]
            pose = model_state.pose[index]
            twist = model_state.twist[index]
            x.append(pose.position.x)
            y.append(pose.position.y)
            theta.append(euler_from_quaternion((pose.orientation.x, 
                                                pose.orientation.y, 
                                                pose.orientation.z, 
                                                pose.orientation.w,))[2])
            correct.append(twist.angular.x)
        x = np.array(x)
        y = np.array(y)
        theta = np.array(theta)
        correct = np.array(correct)
        return x, y, theta, correct

    def get_angle(self,
        x_0: np.ndarray,
        x_1: np.ndarray,
        y_0: np.ndarray,
        y_1: np.ndarray
        ) -> np.ndarray:
        """Returns base angle value between array of two points.

        Args:
            x_0 (np.ndarray): Source x values.
            x_1 (np.ndarray): Positional x values.
            y_0 (np.ndarray): Source y values.
            y_1 (np.ndarray): Positional y values.

        Returns:
            np.ndarray: Angles between array of two points.
        """
        x_diff = x_0 - x_1
        y_diff = y_0 - y_1
        return np.arctan2(y_diff, x_diff) 

    def normalise_scan(self,
        number: float) -> float:
        """ Normalise the scan value with the given minimum and maxiumum range
        Returns:
            float: normalised scan value.
        
        """
        normalised_value = self.MAXIMUM_SCAN_RANGE - (number - self.MINIMUM_SCAN_RANGE) * (self.MAXIMUM_SCAN_RANGE / (self.MAXIMUM_SCAN_RANGE - self.MINIMUM_SCAN_RANGE))
        return normalised_value

    def apply_median_filter(self
        ) -> list:
        # 1. get laser_buffer and transform into list type from queue.
        list_laser_buffer = []
        for i in range(len(self.laser_buffer)):
            list_laser = list(self.laser_buffer[i].queue)
            list_laser_buffer.append(list_laser)

        # 2. Calculate the median of the obtained laser_buffer.
        array_median_scan_ranges = np.median(list_laser_buffer, axis=1)
        list_median_scan_ranges = array_median_scan_ranges.tolist()
        
        return list_median_scan_ranges

    def get_robot_lasers_collisions(self,
        median_scan_ranges: list,
        ) -> tuple:
        """Returns values of all robots lasers and if robots collided.

        Returns:
            tuple: lasers, collisions
        """
        lasers = []
        collisions = [False for i in range(self.robot_count)]
        id_collisions = [0 for i in range(self.robot_count)]
        # each robot
        for i in range(self.robot_count):
            lasers.append(median_scan_ranges[i])
            #scan = self.laser_info_getter[i].get_msg()
            #print(f"scan.ranges: {scan.ranges}")
            # each laser in scan
            for j in range(len(median_scan_ranges[i])):

                # if j == 0:
                #     pass
                # else:
                #lasers[i].append(0)
                if median_scan_ranges[i][j] == float('Inf'):
                    lasers[i][j] = self.normalise_scan(self.MAXIMUM_SCAN_RANGE)
                elif np.isnan(median_scan_ranges[i][j]):
                    lasers[i][j] = self.normalise_scan(self.MAXIMUM_SCAN_RANGE)
                elif median_scan_ranges[i][j] > self.MAXIMUM_SCAN_RANGE:
                    lasers[i][j] = self.normalise_scan(self.MAXIMUM_SCAN_RANGE)
                elif median_scan_ranges[i][j] < self.MINIMUM_SCAN_RANGE:
                    lasers[i][j] = self.normalise_scan(self.MAXIMUM_SCAN_RANGE)
                else:
                    lasers[i][j] = self.normalise_scan(median_scan_ranges[i][j])
            #lasers_deque = deque(lasers[i])
            #print(f"lasers_deque: {lasers_deque}")
            #lasers_deque.rotate(1)
            #print(f"lasers_deque: {lasers_deque}")
            #lasers[i] = list(lasers_deque)
            id_collisions[i] = np.argmax(lasers[i])
            if max(lasers[i]) > self.normalise_scan(self.COLLISION_RANGE):
                
                #print(f"This is a normalised collision value: {self.normalise_scan(self.COLLISION_RANGE)}")
                collisions[i] = True
        lasers = np.array(lasers)
        collisions = np.array(collisions)
        return lasers, collisions, id_collisions

    def put_scan_into_buffer(self
        ) -> None:
        if self.laser_buffer[0].full():    
            for i in range(self.robot_count):
                self.laser_buffer[i].get()

        for i in range(self.robot_count):
                scan = self.laser_info_getter[i].get_msg()
                self.laser_buffer[i].put(scan.ranges)
        return
    def get_current_states(self
        ) -> np.ndarray:
        """Returns starting states.

        Returns:
            np.ndarray: Starting states.
        """
        model_state = self.position_info_getter.get_msg()
        robot_indexes = self.get_robot_indexes_from_model_state(model_state)
        x, y, theta, _ = self.get_positions_from_model_state(model_state, 
                                                             robot_indexes)
        # get current distance to goal
        robot_target_distances = self.get_distance(x, self.x_targets, 
                                                   y, self.y_targets)
        # get current robot angles to targets
        robot_target_angle = self.get_angle(self.x_targets, x, 
                                            self.y_targets, y)
        robot_target_angle_difference = (robot_target_angle - theta - np.pi) % (2 * np.pi) - np.pi
        # get current laser measurements
        self.put_scan_into_buffer()

        list_median_scan_ranges = self.apply_median_filter()
        robot_lasers, robot_collisions, id_collisions = self.get_robot_lasers_collisions(list_median_scan_ranges)
        
        # create state array 
        # = lasers (24), 
        #   action linear x (1), action angular z (1), 
        #   distance to target (1), angle to target (1)
        # = dimension (6, 28)
        s_actions_linear = np.zeros((self.robot_count, 1))
        s_actions_angular = np.zeros((self.robot_count, 1))
        s_robot_target_distances = robot_target_distances.reshape((self.robot_count, 1))
        s_robot_target_angle_difference = robot_target_angle_difference.reshape((self.robot_count, 1))
        assert robot_lasers.shape == (self.robot_count, 24), f'Wrong lasers dimension!: {robot_lasers.shape}'
        assert s_actions_linear.shape == (self.robot_count, 1), 'Wrong action linear dimension!'
        assert s_actions_angular.shape == (self.robot_count, 1), 'Wrong action angular dimension!'
        assert s_robot_target_distances.shape == (self.robot_count, 1), 'Wrong distance to target!'
        assert s_robot_target_angle_difference.shape == (self.robot_count, 1), 'Wrong angle to target!'
        states = np.hstack((robot_lasers, 
                            #s_actions_linear, s_actions_angular, 
                            #s_robot_target_distances, 
                            s_robot_target_angle_difference))
        assert states.shape == (self.robot_count, self.observation_dimension), 'Wrong states dimension!'
        return states
