#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from lego_env_for_edit_2 import Legoenv, Biased_Legoenv
import gymnasium as gym

from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--nmax", type=int,default=14)
parser.add_argument("--ntensor", type=int,default=6)
parser.add_argument("--timesteps", type=int,default=1000)
parser.add_argument("--desc")
args = parser.parse_args()


STABILIZERS = ["IIXXXX", "IIZZZZ", "ZIZZII", "IZZIZI", "IXXXII", "XIXIXI"]
max_legs=args.ntensor*len(STABILIZERS)
num_leg_combinations=max_legs*(max_legs-1)//2
actions_terminated = np.zeros(args.ntensor + num_leg_combinations + 1, dtype=bool)
actions_terminated_2=actions_terminated.copy()
actions_terminated_2[-1]==True

#@profile
def is_valid(state, env, num_max_actions, max_tensors, num_min_actions=0):
    #print(action)
    if state[-1] == 1:
        return actions_terminated
    num_tensor_actions = env.num_tensor_types * env.max_tensors
    num_contractions = env.num_contractions
    # make sure not too many actions are taken
    if num_contractions >= num_max_actions:
        return actions_terminated_2
    
    return env.actions

def mask_fn(env: gym.Env) -> np.ndarray:
    s = env.state
    mask = is_valid(s, env=env, num_max_actions=args.nmax, max_tensors=args.ntensor)
    return np.array(mask)

import time


def main():
    start = time.time()
    env = ActionMasker(Biased_Legoenv(max_tensors=args.ntensor), mask_fn)
    model = MaskablePPO("MlpPolicy", env, learning_rate=0.0003, n_steps=100, verbose=1, gamma=1,ent_coef=.01, tensorboard_log="./lego_tensorboard/")

    checkpoint_callback = CheckpointCallback(save_freq=2048, save_path="./logs/nmax{}nt{}_normerr_{}/checkpoints".format(args.nmax, args.ntensor, args.desc))
    model.learn(total_timesteps=args.timesteps,
                tb_log_name="nmax {} ntensor {} (normalized) {}".format(args.nmax, args.ntensor, args.desc),
                reset_num_timesteps=False,
                callback=checkpoint_callback)
    process_time = time.time() - start

    print(process_time)
if __name__ == '__main__':
    main()
