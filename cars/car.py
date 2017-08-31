import collections
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

NUM_LANES = 4
MAX_SPEED = 4
MAX_Y = 17  # road is quantized into 17 slots in the direction of travel
SELF_Y = 9  # vertical position of our own car
NUM_CARS = 4

MAX_STEPS = 200

LANE_POS_IDX = 0
SPEED_IDX = 1
Y_IDX = 2

ACTION_DO_NOTHING = 0
ACTION_INCREASE_SPEED = 1
ACTION_DECREASE_SPEED = 2
ACTION_CHANGE_LEFT = 3
ACTION_CHANGE_RIGHT = 4
COLLISION_REWARD = -20


class CarEnv(gym.Env):
    """Cars travelling on a road.
    There are 4 cars on the road including ourselves. The road has NUM_LANES number of lanes
    and has MAX_Y slots along its length. At any time, each slot may be occupied by at most one
    car. The frame of reference is fixed so that our car's slot is always the middle slot along the
    length.

    If a car runs off the top end, it reappears at the bottom of its lane.

    Actions available:
        # 0: stay at current location and speed,
        # 1: increase speed if possible,
        # 2: decrease speed if possible,
        # 3: change lane to left if possible,
        # 4: change lane to right if possible,
    """
    metadata = {'render.modes': ['human', 'ansi']}
    def __init__(self, natural=False):
        self.spec = None
        self.steps = 0
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(NUM_LANES),  # my lane number
            spaces.Discrete(MAX_SPEED),  # my speed
            spaces.Discrete(MAX_Y),      # my Y location
            spaces.Discrete(NUM_LANES),  # car 1 lane number
            spaces.Discrete(MAX_SPEED),  # car 1 speed
            spaces.Discrete(MAX_Y),      # car 1 Y location
            spaces.Discrete(NUM_LANES),  # car 2 lane number
            spaces.Discrete(MAX_SPEED),  # car 2 speed
            spaces.Discrete(MAX_Y),      # car 2 Y location
            spaces.Discrete(NUM_LANES),  # car 3 lane number
            spaces.Discrete(MAX_SPEED),  # car 3 speed
            spaces.Discrete(MAX_Y),      # car 3 Y location
        ))
        self._seed()

        # Start the first game
        self._reset()
        self.nA = 5

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        done = self.steps > MAX_STEPS
        if done:
            self._get_obs(), 0, done, {}
        self.steps += 1
        self.update_grid()
        my_cars_forward_motion = self.cars[0][SPEED_IDX]
        reward = my_cars_forward_motion
        for i, car in enumerate(self.cars):
            # Check if the slots ahead are empty. This is a simplistic way of collision detection
            collision = self.is_occupied(car[LANE_POS_IDX], range(car[Y_IDX] + 1, car[Y_IDX] + car[SPEED_IDX] + 1))
            if collision:
                if i == 0: # our car
                    reward = COLLISION_REWARD
                    my_cars_forward_motion = 0
                    new_y = car[Y_IDX]
                else:
                    new_y = car[Y_IDX] - my_cars_forward_motion
            else:
                new_y = (car[Y_IDX] + car[SPEED_IDX] - my_cars_forward_motion)
            new_lane = car[LANE_POS_IDX]
            new_speed = car[SPEED_IDX]
            if new_y >= MAX_Y or new_y < 0:
                new_lane, new_speed = self.np_random.choice(NUM_LANES), self.np_random.choice(MAX_SPEED)
            new_y %= MAX_Y
            self.cars[i] = (new_lane, new_speed, new_y)
        lane, speed, y = self.cars[0]
        new_lane = lane
        if action == ACTION_CHANGE_LEFT:
            new_lane = max(0, lane - 1)
        elif action == ACTION_CHANGE_RIGHT:
            new_lane = min(NUM_LANES - 1, lane + 1)
        elif action == ACTION_INCREASE_SPEED:
            speed = min(MAX_SPEED - 1, speed + 1)
        elif action == ACTION_DECREASE_SPEED:
            speed = max(0, speed - 1)
        if new_lane != lane and self.is_occupied(new_lane, range(y, y + 1)):
            reward = COLLISION_REWARD
            new_lane = lane
        self.cars[0] = new_lane, speed, y
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return np.array([item for sublist in self.cars for item in sublist])

    def update_grid(self):
        self.lane_to_cars = collections.defaultdict(list)
        self.y_to_cars = collections.defaultdict(list)
        for i, car in enumerate(self.cars):
            self.lane_to_cars[car[LANE_POS_IDX]].append(i)
            self.y_to_cars[car[Y_IDX]].append(i)

    def is_occupied(self, lane, y_range):
        for y in y_range:
            occupied = bool(set(self.y_to_cars[y]).intersection(set(self.lane_to_cars[lane])))
            if occupied:
                return True
        return False

    def _render(self, mode='human', close=False):
        if close:
            return
        self.update_grid()
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        for y in range(MAX_Y - 1, -1, -1):
            cars_at_y = self.y_to_cars[y]
            for lane in range(NUM_LANES):
                cars_in_lane = self.lane_to_cars[lane]
                car_indices = set(cars_at_y).intersection(set(cars_in_lane))
                car_strings = [('(%s,%s)' % (carIndex, self.cars[carIndex][SPEED_IDX])) for carIndex in car_indices]
                slot = ','.join(car_strings)
                slot = '%5s' % slot
                outfile.write(slot)
                outfile.write('|')
            outfile.write('\n')
        if mode == 'ansi':
            return outfile

    def init_car(self):
        return self.np_random.choice(NUM_LANES), self.np_random.choice(MAX_SPEED), self.np_random.choice(MAX_Y)

    def _reset(self):
        self.cars = []
        for i in range(NUM_CARS):
            self.cars.append(self.init_car())
        self.cars[0] = (int(NUM_LANES / 2), 0, int(MAX_Y / 2) + 1)
        return self._get_obs()
