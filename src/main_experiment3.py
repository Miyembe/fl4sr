#! /usr/bin/env python3.8
import sys
import os
import time
import rospy
import random
import subprocess
import numpy as np
HOME = os.environ['HOME']
sys.path.append(HOME + '/catkin_ws/src/fl4sr/src')

# This script is for real experiment. 

# COMMANDS
# Commands are saved to command list, which are then then run.
# This is done so the experiment can be restarted after encoutering error.

dict_reward = {'reward_goal': [100.0], 'reward_collision': [-30.0], 'reward_progress': [40.0]}
dict_update_period = {'updatePeriod': [1,3,5]}
dict_seed = {'seed': [601, 602, 603]}
dict_algorithms = {'algorithms': ['FLDDPG']}

COMMAND_LIST = []
for rg, rc, rp in zip(dict_reward['reward_goal'], dict_reward['reward_collision'], dict_reward['reward_progress']):
    for uP in dict_update_period['updatePeriod']:
        for seed in dict_seed['seed']:
            for algo in dict_algorithms['algorithms']:
                COMMAND_LIST.append(['rosrun', 'fl4sr', 'experiment_limit.py', algo, f'--mode={"learn"}', f'--seed={seed}', f'--updatePeriod={uP}', f'--reward_goal={rg}',\
                 f'--reward_collision={rc}',f'--reward_progress={rp}', f'--reward_max_collision={1.0}', f'--list_reward={dict_list["list_reward"]}', '--factor_linear=0.25', f'--discount_factor={0.99}'])

# PRINT
# Print all commands to before their execution.
for i in range(len(COMMAND_LIST)):    
    print(COMMAND_LIST[i])
#exit(0)

# RUN
# Execute each command until the success.
for command in COMMAND_LIST:
    success = False
    restart_command = []
    while not success:
        print(command + restart_command)
        subprocess.run(command + restart_command)

        with open('main.info', 'r') as f:
            result = f.readline()
        open('main.info', 'w').close()
        if result == '':
            print('COMMAND OK')
            success = True
        else:
            restart_command = ['--restart', 'True']

