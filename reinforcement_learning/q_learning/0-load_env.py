#!/usr/bin/env python3
"""
Project Q learning
by Ced
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name, is_slippery=is_slippery)
    return env

