import numpy as np
from copy import deepcopy
import torch
from torch.optim import Adam
from utils import MiniLog


class BaseAgent(object):
  def __init__(self, env):#, env, ReplayBuffer, net, repl_size=10000, action_sp='env'): # action_sp='manual'
    self.obs_dim = env.observation_space.shape[0]
    self.env = env
    #if action_sp == 'env':
    #    action_sp = env.action_space
    #self.buffer = ReplayBuffer(obs_dim, action_sp.shape[0], size=repl_size)
    #self.ac = net(env.observation_space, action_sp)

  def __init_net(self):
      pass
  def store(self, obs, act, rew, next_obs, done):
    pass
  def get_action(self, obs):
    pass

class RMetaAgent(BaseAgent):
  def __init__(self, goal_shape):
    super(RMetaAgent, self).__init__(goal_shape)
  def act(self, obs):
    return np.random.randint(0, 2, size=self.goal_shape)

class DDPGAgent(BaseAgent):
    class ActionSpace:
        def __init__(self, shape):
            self.shape = [shape]
            self.high = np.ones(shape)

    def __init__(self, env, replay_buffer, net, act_shape=None, start_steps=200, update_every=100, iters=None, update_after=1000,
                 repl_size=5000, pi_lr=0.001, q_lr=0.001, batch_size=32, gamma=0.99, polyak=0.995,
                 seed=0, act_noise=0.1, act_limit=None):
        super(DDPGAgent, self).__init__(env)
        self.action_sp = self.env.action_space
        if act_shape:
            self.action_sp = self.ActionSpace(act_shape[-1])
        self.net_args = [net, pi_lr, q_lr, self.env.observation_space, self.action_sp, seed]
        self._init_net(*self.net_args)
        self.act_noise = act_noise
        self.act_dim = env.action_space.shape[0]
        self.act_limit = act_limit or env.action_space.high[0]
        self.buffer = replay_buffer(self.obs_dim, self.act_dim, size=repl_size)
        self.start_steps = start_steps
        self.update_every = update_every
        self.update_after = update_after
        self.iters = iters or update_every


        # Train
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak

        #self.log = MiniLog(100)
        self.t = 0

    def reset_agent(self, seed=0):
        self.net_args[-1] = seed or self.net_args[1]
        self._init_net(*self.net_args)
        self.t = 0
        self.buffer.ptr, self.buffer.size = 0, 0


    def _init_net(self, net, pi_lr, q_lr, obs_sp, act_sp, seed):
        torch.manual_seed(seed)
        np.random.seed(int(seed))
        self.ac = net(obs_sp, act_sp)
        # Create actor-critic module and target networks
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.ac.q.parameters(), lr=q_lr)

    def get_action(self, o, noise_scale=None):
        # noise_scale=0 for test
        if noise_scale is None:
            noise_scale = self.act_noise
        if noise_scale and (self.buffer.ptr < self.start_steps):
            return np.random.rand(self.action_sp.shape[0])
        a = self.ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)


    def store(self, obs, act, rew, next_obs, done):
        self.buffer.store(obs, act, rew, next_obs, done)
        self.t += 1
        #self.log.rput(rew, done)
        if (self.t > self.update_after) and (not self.buffer.ptr % self.update_every):
            self.train()





        # Set up function for computing DDPG Q-loss
    def __compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = self.ac.q(o, a)

        # Bellman backup for Q function
        with torch.no_grad():
            q_pi_targ = self.ac_targ.q(o2, self.ac_targ.pi(o2))
            backup = r + self.gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

        # Set up function for computing DDPG pi loss
    def __compute_loss_pi(self, data):
        o = data['obs']
        q_pi = self.ac.q(o, self.ac.pi(o))
        return -q_pi.mean()



    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def __update(self, data):
        # First run one gradient descent step for Q.
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.__compute_loss_q(data)
        #self.log.lput(loss_q, 'q')
        #self.log.append(loss_q.data.numpy())
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for p in self.ac.q.parameters():
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.__compute_loss_pi(data)
        # print(loss_pi)
        #self.log.lput(loss_pi, 'pi')
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for p in self.ac.q.parameters():
            p.requires_grad = True

        # Record things
        # logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def train(self):

        for _ in range(self.iters):
            batch = self.buffer.sample_batch(self.batch_size)
            self.__update(data=batch)


class HIRO(BaseAgent):
    def __init__(self, env, low_agent_kwargs={}, high_agent_kwargs={},
                 store_delay=200, train_delay=100, step_each=4, ado=None):
        """

        :param env:
        :param store_delay: store after low agent update a little
        :param train_delay: train after store train_delay observations
        :param step_each:
        :param ado: shapes of achive_goal, desire_goal, obs
        """
        self.store_delay = store_delay
        self.train_delay = train_delay
        self.step_each = step_each
        self.done = False
        self.h_reward = 0
        self.t_meta_train = 0
        self.ado = ado or env.action_space.shape * 3  # TODO only for bitflipping
        self.low_agent = DDPGAgent(**low_agent_kwargs)    #env, replay_buf, net=net, start_steps=3000, update_every=50, repl_size=10000)
        self.high_agent = DDPGAgent(**high_agent_kwargs)  # env, replay_buf, net=net, start_steps=3000, update_every=50, repl_size=10000)

    def reset_agent(self, seed=0):
        self.low_agent.reset_agent(seed)
        self.high_agent.reset_agent(seed)

    def get_action(self, o, noise_scale=None):
        # noise_scale=0 for test
        if self.done or self.low_agent.buffer.ptr % self.step_each == 0:
            self.low_goal = self.high_agent.get_action(o, noise_scale)
        o[self.ado[0]:self.ado[0] + self.ado[1]] = self.low_goal
        return self.low_agent.get_action(o, noise_scale)



    def store(self, obs, act, rew, next_obs, done):
        # low agent store
        low_obs = obs[:]
        low_obs[self.ado[0]: self.ado[0] + self.ado[1]] = self.low_goal
        next_low_obs = next_obs[:]
        next_low_obs[self.ado[0]: self.ado[0] + self.ado[1]] = self.low_goal
        low_agent_reward = self.reward_function(low_obs, next_low_obs, self.ado)
        self.low_agent.store(low_obs, act, low_agent_reward, next_low_obs, done)

        # high agent store
        self.done = done
        self.h_reward += rew
        if self.t_meta_train == self.store_delay:
            self.high_agent.update_after = self.low_agent.buffer.ptr + self.train_delay
        if self.t_meta_train > self.store_delay:
            self.high_agent.store(obs, self.low_goal, self.h_reward, next_obs, done)  # act == self.low_goal
            self.h_reward = 0
            self.t_meta_train = 0



    def reward_function(self, obs, next_obs, ado):
        goal = obs[ado[0]:ado[0] + ado[1]]
        o = obs[ado[1]:ado[1] + ado[2]]
        o_next = next_obs[ado[1]:ado[1] + ado[2]]
        return -np.linalg.norm(o + goal - o_next)


if __name__ == '__main__':
    # Test
    import gym
    from replay_buffers import ReplayBuffer
    from nets import MLPActorCritic
    from games import BitFlipping2
    from utils import MiniLog

    env = BitFlipping2(4, 1, False)
    envf = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
    print(env, envf.observation_space.shape[0]/3)
    a = DDPGAgent(envf, ReplayBuffer, net=MLPActorCritic, start_steps=200, update_every=100,
                     repl_size=10000)
    hiro = HIRO(envf, ReplayBuffer)

    obs = envf.reset()

    e = a.get_action(obs)
    print(a.get_action(obs), hiro.get_action(obs))

