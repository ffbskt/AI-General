import gym
from gym import wrappers
import numpy as np
from utils import MiniLog


class BitFlipping2(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, size=3, rad=0.1, discret_space=False, seed=0):
        super(BitFlipping2, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        np.random.seed(seed)
        if discret_space:
            self.action_space = gym.spaces.Discrete(size)
        else:
            # self.action_space = gym.spaces.Box(0, 1, [size])
            self.action_space = gym.spaces.Box(0., 1., shape=(size,), dtype='float32')
        self.size = [size]
        self.rad = rad
        obs = self.reset()
        self.observation_space = gym.spaces.Dict(dict(
            desired_goal=gym.spaces.Box(-10, 10, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=gym.spaces.Box(-10, 10, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=gym.spaces.Box(-10, 10, shape=obs['observation'].shape, dtype='float32'),
        ))

    def step(self, action):
        # Execute one time step within the environment
        self.steps += 1
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.state[action] = not self.state[action]
        else:
            # action = np.argmax(action)
            self.state += action
        obs = {'observation': self.state, 'desired_goal': self.target, 'achieved_goal': self.state}
        reward = self.compute_reward(self.state, self.target)
        done = False
        if self.steps > 2 ** self.size[0] or reward == 0:
            done = True
        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.state = np.random.randint(0, 2, self.size)
            self.target = np.random.randint(0, 2, self.size)
        else:
            self.state = np.random.randint(0, 2, self.size).astype('float32')
            self.target = np.random.randint(0, 2, self.size).astype('float32')
        self.steps = 0
        while (self.state == self.target).any():
            self.target = np.random.randint(0, 2, self.size)
        return {'observation': self.state, 'desired_goal': self.target, 'achieved_goal': self.state}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('state:', self.state, '\n target:', self.target)

    def compute_reward(self, achieved_goal, goal, info=None, rad=0.1):
        # print(achieved_goal, goal, 2 * (np.sum(achieved_goal == goal, -1) < 0) - 1)
        if isinstance(self.action_space, gym.spaces.Discrete):
            return (achieved_goal == goal).all() - 1
        return (np.linalg.norm(achieved_goal - goal) < self.rad) - 1


from gym import ObservationWrapper
from copy import deepcopy
from collections import defaultdict


def reward_function(obs, next_obs, ado):
    goal = obs[ado[0]:ado[0] + ado[1]]
    o = obs[ado[1]:ado[1] + ado[2]]
    o_next = next_obs[ado[1]:ado[1] + ado[2]]
    return -np.linalg.norm(o + goal - o_next)


class Env_add_Meta(gym.Wrapper):
    def __init__(self, env, ado, log=None, meta_agent=None, goal_freq=2,
                 reward_function=reward_function):  # if ado None observation add back
        super(Env_add_Meta, self).__init__(env)
        self.meta_agent = meta_agent
        self.ado = ado
        self.step_n = 0
        self.goal_freq = goal_freq
        self.r_sum = 0
        self.last_obs = None
        self.reward_function = reward_function
        self.o_goal_distant = []
        self.realg_metag_obs = []
        self.log = log or MiniLog(50)

    def step(self, action):
        o, r, d, info = self.env.step(action)
        self.log.rput(r, d)
        self.r_sum += r
        # print('++', o, self.last_obs)
        if d or (self.step_n % self.goal_freq == 0):
            # print('+', o, self.last_obs)

            # if self.last_obs[5] != 0 or self.last_obs[5] != 1: print('ru', self.last_obs, o), dd
            # print(meta_agent.buffer.obs_buf.shape)
            self.meta_agent.store(self.last_meta_obs, self.desire_goal, self.r_sum, o, d)
            self.last_meta_obs = deepcopy(o)
            self.desire_goal = self.meta_agent.get_action(o)
            self.r_sum = 0
        self.step_n += 1
        self.o_goal_distant.append(-np.linalg.norm(self.desire_goal - o[self.ado[1]:self.ado[1] + self.ado[2]]))
        o[self.ado[0]:self.ado[0] + self.ado[1]] = self.desire_goal
        rew = self.reward_function(self.last_low_obs, o, self.ado)
        self.last_low_obs = deepcopy(o)
        return o, rew, d, info

    def reset(self):
        self.step_n = 1
        o = self.env.reset()
        self.desire_goal = self.meta_agent.get_action(o)  # noise ?
        self.last_meta_obs = deepcopy(o)
        self.r_sum = 0
        o[self.ado[0]:self.ado[0] + self.ado[1]] = self.desire_goal
        self.last_low_obs = deepcopy(o)
        return o