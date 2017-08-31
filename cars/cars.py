import gym
import numpy as np
import sys

if "../" not in sys.path:
  sys.path.append("../")

from lib.envs.car import CarEnv

env = CarEnv()
VALID_ACTIONS = [0, 1, 2, 3, 4]

while True:
  env.render()
  action = int(input(""))
  next_state, reward, done, _ = env.step(VALID_ACTIONS[action])
  print(next_state, reward, done)


