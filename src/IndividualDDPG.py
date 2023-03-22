#! /usr/bin/env python3.8
import sys
import os
import torch
HOME = os.environ['HOME']
#sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')
import numpy as np
import time
import pickle
import json
import matplotlib.pyplot as plt
import rospkg
from Environment import Environment
#from Environment_eval import Environment_eval
#from environment_real import RealEnvironment
from worlds import World
from DDPG import DDPG
from buffers import BasicBuffer, PrioritizedExperienceReplayBuffer, Transition
import datetime
import random
import matplotlib
import matplotlib.pyplot as plt
import time
import cv2
from PIL import Image
import imageio
from celluloid import Camera


def write_video_PIL(frames, file_name, fps=30):

    imageio.mimwrite(uri=file_name, ims=frames, fps=fps, format=".gif")


def write_frame_number(frame, text):
    frame = cv2.UMat(frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1
    font_color = (0, 0, 255)  # BGR format for red color

    # Define the text to be added and its position
    text_position = (2, 10)

    # Add the text to the image
    cv2.putText(
        frame, text, text_position, font, font_scale, font_color, font_thickness
    )

    new_frame = frame.get()
    return new_frame





class IndividualDDPG():
    """Individial DDPG algorithm. Serves as basis for other algorithms.
    """

    def __init__(self,
        episode_count: int,
        episode_step_count: int,
        world: World,  
        env = 'Environment', 
        reward_goal: float = 100.0,
        reward_collision: float = -30.0,
        reward_progress: float = 40.0,
        reward_max_collision: float = 3.0,
        list_reward: int = 1,
        factor_linear: float = 0.25,
        factor_angular: float = 1.0,
        discount_factor: float = 0.99,
        num_parameters: int = 128,
        learning_rate: float = 0.001,
        batch_size: int = 512,
        buffer_type: str = 'BasicBuffer',
        is_progress: bool = False,
        name=None,
        model_name: str = ""
        ) -> None:
        """Initialize class and whole experiment.
        Args:
            episode_count (int): ...
            episode_step_count (int): ...
            world (World): contains information about experiment characteristics
            name (str, optional): Name of used method. Defaults to None.
        """
        print(f"INSIDE IndividualDDPG: episode_count: {episode_count}, episode_step_count: {episode_step_count}. world: {world}, env: {env}, reward_goal: {reward_goal}, reward_collision: {reward_collision}, reward_progress: {reward_progress}, reward_max_collision: {reward_max_collision}, list_reward: {list_reward}, factor_linear: {factor_linear}, factor_angular: {factor_angular}, discount_factor: {discount_factor}, is_progress: {is_progress}, method: {name}")
        self.rospack = rospkg.RosPack()
        self.pkg_path = self.rospack.get_path('fl4sr')
        
        # global like variables
        self.TIME_TRAIN = 5
        self.TIME_TARGET = 5
        self.EPISODE_UPDATE = True
        self.TIME_UPDATE = 2
        self.TIME_LOGGER = 16
        self.TIME_SAVE = 25
        self.ANIMATION_UPDATE = 64
        # random actions
        self.EPSILON = 0.9
        self.EPSILON_DECAY = 0.9997
        # init experiment and error values
        self.episode_count = episode_count
        self.episode_step_count = episode_step_count
        self.episode_error = 0
        self.episode_step_error = 0
        # init some world values
        self.robot_count = world.robot_count
        # Parameters for experiment
        self.reward_goal= reward_goal
        self.reward_collision= reward_collision
        self.reward_progress= reward_progress
        self.list_reward = list_reward
        self.reward_max_collision = reward_max_collision
        self.factor_linear= factor_linear
        self.factor_angular= factor_angular
        self.is_progress=is_progress
        self.model_name = model_name
        self.discount_factor = discount_factor
        self.num_parameters = num_parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # init Environment and dimensions
        self.world = world
        self.env = env
        self.init_environment()
        # init buffers and agents
        if buffer_type == "BasicBuffer":
            self.BUFFER_TYPE = BasicBuffer
        elif buffer_type == "PER":
            self.BUFFER_TYPE = PrioritizedExperienceReplayBuffer
        if not hasattr(self, 'BUFFER_SIZE'):
            self.BUFFER_SIZE = 30000
        self.buffers = self.init_buffers()
        self.agents = self.init_agents()
        # loggers
        if not hasattr(self, 'NAME'):
            if not model_name is "":
                self.NAME = model_name
            else:
                if name is not None:
                    self.NAME = name
                else:
                    self.NAME = 'IDDPG'
        print(f"NAME = {self.NAME}")

        # actor critic output related
        self.range_x = [0, 1]
        self.range_y = [-1, 1]
        self.num_points = 10

        self.init_data()
        # debugging
        self.debug = False
        print(self.buffers)
        print(self.agents)
        # paths
        self.init_paths()

        

    

        return

    def init_environment(self
        ) -> None:
        """Initializes environment.
        """
        if self.env == 'Environment':
            self.environment = Environment(self.world, self.reward_goal,
        self.reward_collision,
        self.reward_progress,
        self.reward_max_collision,
        self.list_reward,
        self.factor_linear,
        self.factor_angular, self.is_progress)
        elif self.env == 'Environment_eval':
            self.environment = Environment_eval(self.world, self.model_name, self.reward_goal,
        self.reward_collision,
        self.reward_progress,
        self.reward_max_collision,
        self.factor_linear,
        self.factor_angular, self.is_progress)
        elif self.env == 'RealEnvironment':
            self.environment = RealEnvironment(self.world)
        else: raise Exception(f"No Environment named {self.env} is available.")
        self.observation_dimension = self.environment.observation_dimension
        self.action_dimension = self.environment.action_dimension
        return

    def init_buffers(self
        ) -> list:
        """Creates list with buffers.
        Returns:
            list: Buffers list.
        """
        return [self.BUFFER_TYPE(self.BUFFER_SIZE) 
                for i in range(self.robot_count)]

    def init_agents(self
        ) -> list:
        """Creates list with agents.
        Returns:
            list: Agents list.
        """
        return [DDPG(self.buffers[i], 
                     self.observation_dimension, 
                     self.action_dimension,
                     self.discount_factor,
                     self.num_parameters,
                     self.learning_rate,
                     self.batch_size) 
                for i in range(self.robot_count)]

    def init_paths(self):
        """Initializes and creates file system for saving obtained information.
        """
        path_data = self.pkg_path + '/src/data'#HOME + '/catkin_ws/src/fl4sr/src/data'

        if not self.model_name is "":
            name_run = self.NAME
        else:
            name_run = self.NAME + '-' + time.strftime("%Y%m%d-%H%M%S")
        self.name_run = name_run
        self.path_run = path_data + '/' + name_run
        self.path_weights = self.path_run + '/weights'
        self.path_log = self.path_run + '/log'
        if not os.path.exists(self.path_weights):
            os.makedirs(self.path_weights, exist_ok=True)
        if not os.path.exists(self.path_log):
            os.makedirs(self.path_log, exist_ok=True)
        return

    def init_data(self
        ) -> None:
        """Initializes data containers.
        """
        self.average_rewards = np.zeros((self.episode_count, self.robot_count))
        self.average_policy_loss = np.zeros((self.episode_count, self.robot_count))
        self.average_critic_loss = np.zeros((self.episode_count, self.robot_count))
        self.robots_succeeded_once = np.zeros((self.episode_count, self.robot_count), dtype=bool)        
        self.robots_finished = np.zeros((self.episode_count, self.robot_count), dtype=bool)
        self.list_critic_frames = [[] for _ in range(self.robot_count)]
        self.list_policy_frames = [[] for _ in range(self.robot_count)]
        self.data = []      
        return

    def init_data_test(self
        ) -> None:
        """Initializes data containers for evaluation.
        """
        self.robots_succeeded_once = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)        
        self.robots_finished = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)
        self.data = []      
        return

    def init_data_eval_list(self
        ) -> None:
        """Initializes data containers for evaluation.
        """
        self.list_robot_succeeded = []
        self.list_arrival_time = []
        self.list_traj_eff = []
        return

    def init_data_real(self
        ) -> None:
        """Initializes data containers for evaluation.
        """
        self.robots_succeeded_once = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)        
        self.robots_finished = np.zeros((self.episode_step_count, self.robot_count), dtype=bool)
        self.data = []
        self.exp_time = {}
        return

    def terminate_environment(self):
        """Sets Environment to None. 
        Used before saving whole class when error is encountered.
        """
        self.environment = None
        return

    def run(self
        ) -> tuple:
        """Runs learning experiment.
        Returns:
            tuple: bool success (no errors encoutered), error episode, error step
        """
        # before start
        self.parameters_save()
        self.print_starting_info()
        total_rewards = np.zeros(self.robot_count)
        total_policy_losses = np.zeros(self.robot_count)
        total_critic_losses = np.zeros(self.robot_count)

        # list_critic_frames = [[] for _ in range(self.robot_count)]
        # list_actor_frames = [[] for _ in range(self.robot_count)]

        print(f"self.env: {self.env}")
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.environment.reset()
            print(f"environment reset.")
            current_states = self.environment.get_current_states()
            data_total_rewards = np.zeros(self.robot_count)
            data_total_policy_losses = np.zeros(self.robot_count)
            data_total_critic_losses = np.zeros(self.robot_count)
            if self.episode_error != episode:
                self.episode_step_error = 0
            for step in range(self.episode_step_error, self.episode_step_count):
                actions = self.agents_actions(current_states)
                actions = self.actions_add_random(actions, episode)
                # perform step
                new_states, rewards, robots_finished, robots_succeeded_once, error, _ = self.environment.step(actions)
                total_rewards += rewards
                data_total_rewards += rewards
                self.buffers_save_transitions(current_states, actions, rewards, new_states, robots_finished)
                # train
                if step % self.TIME_TRAIN == 0:
                    array_policy_loss, array_critic_loss = self.agents_train()
                    data_total_policy_losses += array_policy_loss
                    data_total_critic_losses += array_critic_loss
                    
                # if step % self.ANIMATION_UPDATE == 0:
                #     for i in range(self.robot_count):
                #         array_q_values = self.agent_dist_critic(current_states, i)
                #         self.list_critic_frames[i].append(array_q_values)
                #         array_policy_output = self.agent_dist_actor(i)
                #         self.list_policy_frames[i].append(array_policy_output)
                #         # array_linear_output = array_policy_output[:][1]
                #         # array_actor_output = array_policy_output[:][0]
                #         #print(f"policy output for agent {i}: {array_policy_output}")

                # update target
                if step % self.TIME_TARGET == 0:
                    self.agents_target()
                # step federated update
                if (not self.EPISODE_UPDATE) and step % self.TIME_UPDATE == 0:
                    print('UPDATE')
                    mean_rewards = total_rewards / self.TIME_UPDATE
                    self.agents_update(mean_rewards)
                    total_rewards = np.zeros(self.robot_count)
                # print info
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                    print(actions)
                current_states = new_states
            self.data_collect(episode, data_total_rewards, data_total_policy_losses, data_total_critic_losses, robots_succeeded_once)
            # print info
            print('Average episode rewards: {}'.format(self.average_rewards[episode]))
            # episode federated update
            if self.EPISODE_UPDATE and episode % self.TIME_UPDATE == 0:
                print('UPDATE')
                mean_rewards = total_rewards / (self.TIME_UPDATE * self.episode_step_count)
                self.agents_update(mean_rewards)
                total_rewards = np.zeros(self.robot_count)
            # save data
            if episode % self.TIME_SAVE == 0:
                self.agents_save(episode)
                self.data_save(episode)
        self.environment.reset()
        self.agents_save()
        self.data_save()
        self.plot_save(self.path_log)
        return True, None, None

    def test(self
        ) -> tuple:
        """Runs evaluation experiment.
        Returns:
            tuple: bool success (no errors encountered), error episode, error step
        """
        # before start
        self.init_data_test()
        self.init_data_eval_list()
        self.parameters_save()
        self.print_starting_info(False)
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.environment.reset()
            self.init_data_test()
            current_states = self.environment.get_current_states()
            if self.episode_error != episode:
                self.episode_step_error = 0
            for step in range(0, self.episode_step_count):
                actions = self.agents_actions(current_states)
                new_states, rewards, robots_finished, robots_succeeded_once, error, data = self.environment.step(actions)
                if error:
                    self.episode_error = episode
                    self.episode_step_error = step
                    print('ERROR: DDPG: Death robot detected during {}.{}'.format(episode, step))
                    return False, episode, step
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                    print(actions)
                current_states = new_states
                self.data_collect_test(step, robots_finished, robots_succeeded_once, data)
                if np.all(robots_finished):
                    break
            print('Robots succeded once: {}'.format(robots_succeeded_once))
            self.data_save_test(episode)
        self.environment.reset()
        self.data_collect_eval_list(self.environment.list_robot_succeeded, self.environment.list_arrival_time, self.environment.list_traj_eff)
        self.data_save_eval_list()
        return True, None, None

    def test_real(self
        ) -> tuple:
        """Runs evaluation experiment.
        Returns:
            tuple: bool success (no errors encountered), error episode, error step
        """
        # before start
        self.init_data_real()
        self.parameters_save()
        self.print_starting_info(False)
        # epizode loop
        for episode in range(self.episode_error, self.episode_count):
            self.environment.reset()
            self.init_data_real()
            current_states = self.environment.get_current_states()
            if self.episode_error != episode:
                self.episode_step_error = 0
            print(f"timer started")
            start_time = time.time()
            for step in range(0, self.episode_step_count):
                print(f"step started: {step}")
                actions = self.agents_actions(current_states)
                new_states, rewards, robots_finished, robots_succeeded_once, error, data = self.environment.step(actions)
                print(f"step: {step}, states: {current_states}, actions: {actions}, rewards: {rewards}, new_states: {new_states}, robot_finished: {robots_finished}")
                if error:
                    self.episode_error = episode
                    self.episode_step_error = step
                    print('ERROR: DDPG: Death robot detected during {}.{}'.format(episode, step))
                    return False, episode, step
                if step % self.TIME_LOGGER == 0:
                    print('{}.{}'.format(episode, step))
                current_states = new_states
                self.data_collect_test(step, robots_finished, robots_succeeded_once, data)
                if np.any(robots_finished):
                    break
            self.exp_time[f'{episode}'] = time.time()-start_time
            print('Robots succeded once: {}'.format(robots_succeeded_once))
            self.data_save_real(episode)
        return True, None, None

    

    def agents_actions(self,
        states: np.ndarray,
        ) -> np.ndarray:
        """Get actions of all agents.
        Args:
            states (np.ndarray): ...
        Returns:
            np.ndarray: actions
        """
        actions = []
        for i in range(self.robot_count):
            actions.append(self.agents[i].select_action(states[i]))
        actions = np.array(actions)
        return actions
    
    def agent_dist_critic(self,
        states: np.ndarray,
        id: int
        ):
        
        actions = []
        num_samples = self.num_points
        range_linear_action = self.range_x
        range_linear_splice = (range_linear_action[1] - range_linear_action[0])/num_samples
        range_angular_action = self.range_y
        range_angular_splice = (range_angular_action[1] - range_angular_action[0])/num_samples

        
        array_linear = np.linspace(range_linear_action[0], range_linear_action[1], num_samples)
        array_angular = np.linspace(range_angular_action[0], range_angular_action[1], num_samples)
        array_state_action = torch.zeros([num_samples, num_samples])
        critic = self.agents[id].critic
        state = states[id]

        #array_linear, array_angular = np.mgrid[range_linear_action[0]:range_linear_action[1]:range_linear_splice,
        #                                        range_angular_action[0]:range_angular_action[1]:range_angular_splice]
    
        list_state = []
        list_action = []
        for i, linear in enumerate(array_linear):
            for j, angular in enumerate(array_angular):
                if isinstance(state, np.ndarray):
                    list_state.append(state) 
                    #state = torch.from_numpy(state).type(torch.cuda.FloatTensor)
                action = [angular, linear]
                list_action.append(action)
                #action = torch.Tensor([angular, linear]).type(torch.cuda.FloatTensor)
                #state_action_t = [state, action]
                #array_state_action[i, j] = state_action_t
        tensor_state = torch.Tensor(list_state).cuda()
        tensor_action = torch.Tensor(list_action).cuda()
        tuple_state_action = (tensor_state, tensor_action)
        #tensor_state_action = array_state_action.cuda()
        #print(f"tensor_state_action: {tensor_state_action}")
        tensor_critic = critic(tuple_state_action)
        array_critic = tensor_critic.detach().cpu().numpy()
        return array_critic

    def agent_dist_actor(self,
        id: int
        ):
        range_angle_diff = self.range_y
        num_samples = self.num_points
        array_angle_diff = np.linspace(range_angle_diff[0], range_angle_diff[1], num_samples)
        list_states = []
        list_actor = []
        actor = self.agents[id].actor
        for i, state in enumerate(array_angle_diff):
            list_obs = np.zeros(24)
            list_obs = list(list_obs)
            list_obs.append(state)
            #array_obs = np.array(list_obs)
            # if isinstance(array_obs, np.ndarray):
            #     array_obs = torch.from_numpy(array_obs).type(torch.cuda.FloatTensor)
            list_states.append(list_obs)
        array_states = np.array(list_states)
        tensor_states = torch.from_numpy(array_states).type(torch.cuda.FloatTensor)
        policy_output = actor(tensor_states)
        list_actor.append(policy_output.detach().cpu().numpy())
        array_actor = np.array(list_actor)
        print(f"array actor: {array_actor}, shape: {array_actor.shape}")
        return array_actor

        
    def buffers_save_transitions(self, 
        s: np.ndarray, 
        a: np.ndarray, 
        r: np.ndarray, 
        s_: np.ndarray, 
        f: np.ndarray
        ) -> None:
        """Save transitions to buffers.
        Args as described in thesis, only "D" is named as "f".
        Args:
            s (np.ndarray): ...
            a (np.ndarray): ...
            r (np.ndarray): ...
            s_ (np.ndarray): ...
            f (np.ndarray): ...
        """
        for i in range(self.robot_count):
            self.buffers[i].add([Transition(s[i], a[i], r[i], s_[i], f[i])])
        return

    def actions_add_random(self, 
        actions: np.ndarray,
        episode: int
        ) -> np.ndarray:
        """Add random actions.
        Args:
            actions (np.ndarray): ...
            episode (int): not used
        Returns:
            np.ndarray: actions possibly with some actions randomized
        """
        # get current actions
        angles_a = actions[:, 0]
        linears_a = actions[:, 1]
        print(f"prev_angles_a: {angles_a}")
        print(f"prev_linears_a: {linears_a}")
        # where to use randoms and generate them
        randoms = np.random.uniform(0, 1, self.robot_count)
        use_randoms = np.where(randoms < self.EPSILON, 1, 0)
        angles_r = np.random.uniform(-1, 1, self.robot_count)
        linears_r = np.random.uniform(0, 1, self.robot_count)
        # add randoms and clip
        angles = (1 - use_randoms) * angles_a + use_randoms * angles_r
        linears = (1 - use_randoms) * linears_a + use_randoms * linears_r
        angles = np.clip(angles, -1, 1)
        linears = np.clip(linears, 0, 1)
        print(f"cur_angles_a: {angles_a}")
        print(f"cur_linears_a: {linears_a}")
        # combine new actions
        new_actions = np.array((angles, linears)).T
        # update random rate
        self.EPSILON *= self.EPSILON_DECAY
        return new_actions


    def agents_train(self):
        """Train all agents.
        """
        list_policy_loss = []
        list_critic_loss = []
        for agent in self.agents:
            tuple_loss = agent.train()
            list_policy_loss.append(tuple_loss[0])
            list_critic_loss.append(tuple_loss[1])
        array_policy_loss = np.array(list_policy_loss)
        array_critic_loss = np.array(list_critic_loss)
        return array_policy_loss, array_critic_loss

    def agents_target(self):
        """Update target networks of all agents.
        """
        for agent in self.agents:
            agent.update_targets() 
        return

    def agents_update(self, rewards):
        """Update parameters of agents.
        (Obviously empty for IDDPG, SEDDPG, and SNDDPG)
        Args:
            rewards (np.array): average rewards obtained by agents between updates 
        """
        return

    def agents_save(self, 
        episode:int=None
        ) -> None:
        """Save weights of agents.
        Args:
            episode (int, optional): Current episode for file naming. Defaults to None.
        """
        if episode is None:
            episode = 'final'
        for i in range(len(self.agents)):
            self.agents[i].weights_save(self.path_weights + '/actor-{}-{}.pkl'
                                                            .format(episode, i),
                                        self.path_weights + '/critic-{}-{}.pkl'
                                                            .format(episode, i))
        return

    def agents_load(self, 
        paths_actor: list, 
        paths_critic: list
        ) -> None:
        """Load weights of agents.
        Args:
            paths_actor (list): list of paths to actors
            paths_critic (list): list of paths to critics
        """
        assert len(paths_actor) == len(paths_critic) == len(self.agents), 'Wrong load size!'
        for i in range(len(self.agents)):
            self.agents[i].weights_load(paths_actor[i], 
                                        paths_critic[i])
            self.agents[i].update_targets()
        return

    def data_collect(self, 
        episode,
        total_rewards,
        total_policy_loss, 
        total_critic_loss,
        robots_succeeded_once
        ) -> None:
        """Collect data from learning experiments.
        Args:
            episode (int): ...
            total_rewards (np.ndarray): ...
            robots_succeeded_once (np.ndarray): ...
        """
        self.average_rewards[episode] = total_rewards / self.episode_step_count
        self.average_policy_loss[episode] = total_policy_loss / (self.episode_step_count/self.TIME_TRAIN)
        self.average_critic_loss[episode] = total_critic_loss / (self.episode_step_count/self.TIME_TRAIN)
        self.robots_succeeded_once[episode] = robots_succeeded_once
        if episode != 0:
            same_indexes = np.where(self.average_rewards[episode-1] == self.average_rewards[episode])[0]
            if len(same_indexes) > 0:
                self.debug = True
                print('ERROR: Suspicious behaviour discovered, repeated fitness for robots {}'.format(same_indexes))
        return

    def data_collect_test(self, 
        step,
        robots_finished, 
        robots_succeeded_once,
        data
        ) -> None:
        """Collect data from evaluating experiments.
        Args:
            step (int): ...
            robots_finished (np.ndarray): ...
            robots_succeeded_once (np.ndarray): ...
            data (_type_): additional collected information for each step
        """
        self.robots_finished[step] = robots_finished
        self.robots_succeeded_once[step] = robots_succeeded_once
        self.data.append(data)
        return

    def data_collect_eval_list(self,
        list_robot_succeeded,
        list_arrival_time,
        list_traj_eff
        ) -> None:
        """Collect list of data for fixed repetition of experiment.
        Args:
            list_robot_succeeded (list): list of successful episode for each agent for defined amount of runs
            list_arrival_time (list): list of arrival time of each agent for n runs when successful.
            list_traj_eff (list): list of traj eff of each agent for n runs when successful.
        """
        self.list_robot_succeeded = list_robot_succeeded
        self.list_arrival_time = list_arrival_time
        self.list_traj_eff = list_traj_eff

        return

    

    def data_save(self, 
        episode:int=None
        ) -> None:
        """Save collected data form learning.
        Args:
            episode (int, optional): ... . Defaults to None.
        """
        np.save(self.path_log + '/rewards'.format(), 
                self.average_rewards)
        np.save(self.path_log + '/policy_loss'.format(), 
                self.average_policy_loss)
        np.save(self.path_log + '/critic_loss'.format(), 
                self.average_critic_loss)
        np.save(self.path_log + '/succeded'.format(),
                self.robots_succeeded_once)

        return

    def save_actor_critic_output_animation(self,
        list_critic_frames: list,
        list_actor_frames: list,
        range_x: list,
        range_y: list,
        num_points: int,
        file_name: str,
        ):
        y, x = np.meshgrid(np.linspace(range_y[0], range_y[1], num_points), np.linspace(range_x[0], range_x[1], num_points))
        fig, axs = plt.subplots(3, dpi=100)
        camera = Camera(fig)
        list_actor_frames = np.array(list_actor_frames).squeeze()
        #print(f"list_critic_frames: {list_critic_frames}")
        x_actor = np.linspace(-1, 1, num_points)
        num_frame = len(list_critic_frames)
        for i in range(num_frame):
            z = list_critic_frames[i]
            z = z.reshape(num_points, num_points)
            z = z[:-1, :-1]
            z_min, z_max= -np.abs(z).max(), np.abs(z).max()
            c = axs[0].pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
            axs[0].set_title('critic_output', fontsize=10)
            axs[0].set_xlabel('Linear Velocity', fontsize=8)
            axs[0].set_ylabel('Angular Velocity', fontsize=8)
            axs[0].axis([x.min(), x.max(), y.min(), y.max()])
            #axs[0].colorbar(c, ax=ax)
            #axs[1].plot(xlist_linear_frames[i][np.newaxis, :], cmap="plasma", aspect="auto")
            #axs[2].plot(list_angular_frames[i][np.newaxis, :], cmap="plasma", aspect="auto")
            axs[1].set_title('linear velocity output', fontsize=10)
            axs[1].plot(x_actor, list_actor_frames[i][:, 1])
            axs[1].set_xlabel('Normalised Angle Difference (-1 - 1)', fontsize=8)
            axs[1].set_ylabel('Linear Velocity [0 - 1]', fontsize=8)
            axs[1].set_ylim(-0.2, 1.2)
            axs[1].text(0.7, 0.5, f'Episode: {self.ANIMATION_UPDATE * i // self.episode_step_count}', style ='italic',
            fontsize = 12, color ="green")
            axs[1].text(0.7, 0.2, f'Step {self.ANIMATION_UPDATE * i % self.episode_step_count}', style ='italic',
            fontsize = 12, color ="green")
            axs[2].set_title('angular velocity output', fontsize=10)
            axs[2].plot(x_actor, list_actor_frames[i][:, 0])
            axs[2].set_xlabel('Normalised Angle Difference (-1 - 1)' , fontsize=8)
            axs[2].set_ylabel('Angular Velocity [-1 - 1]', fontsize=8)
            axs[2].set_ylim(-1.2, 1.2)

            camera.snap()


        animation = camera.animate()
        animation.save(f'{file_name}.gif', writer ='imagemagick')

    def plot_save(self,
        log_path:str
        ) -> None:
        """Save training curve plot of rewards.npy .
        Args:
            log_path (str): ... . Defaults to the path of rewards.npy saved by data_save.
        """ 
        figure, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=300)
        #plt.xlabel('Episodes')
        #plt.ylabel('Rewards')
        values = self.average_rewards.T
        num_time_steps = values.T.shape[0]
        list_time_steps = [i for i in range(num_time_steps)]
        num_agents = 4
        list_agents = [f"Robot {i}" for i in range(num_agents)]
        #print(f"shape of values_std: {np.array(values_std).shape}")
        ax.plot(list_time_steps, values.T)
        #for k in range(len(list_algorithm)):
        #print(values.T[k])
        #print(values_std.T[k])
        #axs[i].fill_between(list_time_steps, values[k]-values_std[k], values[k]+values_std[k], alpha=0.5)
        ax.set_title(f'Experiment name: {self.name_run}')
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Rewards')
        ax.set_ylim(0, 15)
        ax.legend(list_agents)
        figure_name = self.name_run +'.png'
        figure_path = os.path.join(self.path_log, figure_name)
        figure.savefig(figure_path)

        # for i in range(self.robot_count):
        #     path_actor_critic_i = os.path.join(self.path_log, f'actor_critic_output_{i}')
        #     self.save_actor_critic_output_animation(
        #         self.list_critic_frames[i],
        #         self.list_policy_frames[i],
        #         self.range_x,
        #         self.range_y,
        #         self.num_points,
        #         path_actor_critic_i)

    def args_save(self,
        args
        ) -> None:
        """Save Argparse used for this experiment for post-experiment use.
        """
        args_file_name = "args.json"
        args_path = os.path.join(self.path_log, args_file_name)
        with open(args_path, 'w') as f:
            json.dump(str(args), f)
        return

        


    def data_save_test(self, 
        episode:int=None
        ) -> None:
        """Save collected data from evaluating.
        Args:
            episode (int, optional): ... . Defaults to None.
        """
        np.save(self.path_log + '/finished-{}'.format(episode), 
                self.robots_finished)
        np.save(self.path_log + '/succeded-{}'.format(episode),
                self.robots_succeeded_once)
        with open(self.path_log + '/data-{}.pkl'.format(episode), 'wb') as f:
            pickle.dump(self.data, f)
        return

    def data_save_eval_list(self
        ) -> None:
        """Save collected data from evaluating.
        Args:
            None.
            TODO I also want to compute average robot_succeeded, arrival_time and traj_eff....
        """
        np.save(self.path_log + '/list_robot_succeeded', 
                self.list_robot_succeeded)
        np.save(self.path_log + '/list_arrival_time', 
                self.list_arrival_time)
        np.save(self.path_log + '/list_traj_eff', 
                self.list_traj_eff)
        
        return
        
        
    def data_save_real(self, 
        episode:int=None
        ) -> None:
        """Save collected data from evaluating.
        Args:
            episode (int, optional): ... . Defaults to None.
        """
        np.save(self.path_log + '/finished-{}'.format(episode), 
                self.robots_finished)
        np.save(self.path_log + '/succeded-{}'.format(episode),
                self.robots_succeeded_once)
        np.save(self.path_log + '/exp_time-{}'.format(episode),
                self.exp_time)
        with open(self.path_log + '/data-{}.pkl'.format(episode), 'wb') as f:
            pickle.dump(self.data, f)
        return
    
    
    def parameters_save(self
        ) -> None:
        """Save used parameters to file.
        """
        parameters = {}
        # method parameters
        parameters['NAME'] = self.NAME
        parameters['robot_count'] = self.robot_count
        parameters['observation_dimension'] = self.observation_dimension
        parameters['action_dimension'] = self.action_dimension
        parameters['episode_count'] = self.episode_count
        parameters['episode_step_count'] = self.episode_step_count        
        parameters['TIME_TRAIN'] = self.TIME_TRAIN
        parameters['TIME_TARGET'] = self.TIME_TARGET
        parameters['TIME_UPDATE'] = self.TIME_UPDATE
        parameters['EPSILON'] = self.EPSILON
        parameters['EPSILON_DECAY'] = self.EPSILON_DECAY
        parameters['buffer_count'] = len(self.buffers)
        parameters['BUFFER_SIZE'] = self.BUFFER_SIZE
        parameters['agent_count'] = len(self.agents)
        if self.NAME == 'FederatedLearningDDPG':
            parameters['TAU_UPDATE'] = self.TAU
        # ddpg parameters
        parameters['ACTOR_HIDDEN_LAYERS'] = self.agents[0].ACTOR_HIDDEN_LAYERS
        parameters['CRITIC_HIDDEN_LAYERS'] = self.agents[0].CRITIC_HIDDEN_LAYERS
        parameters['LEARNING_RATE_ACTOR'] = self.agents[0].LEARNING_RATE_ACTOR
        parameters['LEARNING_RATE_CRITIC'] = self.agents[0].LEARNING_RATE_CRITIC
        parameters['BATCH_SIZE'] = self.agents[0].BATCH_SIZE
        parameters['GAMMA'] = self.agents[0].GAMMA
        parameters['TAU_TARGET'] = self.agents[0].RHO
        # Environmental params
        parameters['COLLISION_RANGE'] = self.environment.COLLISION_RANGE
        parameters['GOAL_RANGE'] = self.environment.GOAL_RANGE
        parameters['PROGRESS_REWARD_FACTOR'] = self.environment.PROGRESS_REWARD_FACTOR
        parameters['REWARD_GOAL'] = self.environment.REWARD_GOAL
        parameters['REWARD_COLLISION'] = self.environment.REWARD_COLLISION
        # save parameters
        with open(self.path_log + '/parameters.pkl', 'wb+') as f:
            pickle.dump(parameters, f)
        return

    def print_starting_info(self, 
        training: bool=True
        ) -> None:
        """Print staring information about experiment.
        Args:
            training (bool, optional): ... . Defaults to True.
        """
        print('{}'.format(self.NAME))
        print('----------------')
        print('Episodes = {}'.format(self.episode_count))
        print('Steps per episode = {}'.format(self.episode_step_count))
        print('Running robots = {}'.format(self.world.robot_alives))
        print('Training = {}'.format(training))
        print('Buffers = {}'.format(len(self.buffers)))
        print('Buffer size = {}'.format(self.BUFFER_SIZE))
        print('Agents = {}'.format(len(self.agents)))
        print('----------------')
        return
