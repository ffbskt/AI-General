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
    def __init__(self, env, ReplayBuffer, net, start_steps=200, update_every=100, iters=None, update_after=1000,
                 repl_size=5000, pi_lr=0.001, q_lr=0.001, batch_size=32, gamma=0.99, polyak=0.995,
                 seed=0, act_noise=0.1):
        super(DDPGAgent, self).__init__(env)
        self.act_noise = act_noise
        self.action_sp = self.env.action_space
        self.act_dim = env.action_space.shape[0]
        self.act_limit = env.action_space.high[0]
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, size=repl_size)
        self.start_steps = start_steps
        self.update_every = update_every
        self.update_after = update_after
        self.iters = iters or update_every
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._init_net(net, pi_lr, q_lr, self.env.observation_space, self.action_sp)

        # Train
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak

        self.log = MiniLog(100)
        self.t = 0

    def _init_net(self, net, pi_lr, q_lr, obs_sp, act_sp):
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
        self.log.rput(rew, done)
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
        self.log.lput(loss_q, 'q')
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
        self.log.lput(loss_pi, 'pi')
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

class MetaAgent(DDPGAgent):
    class ActionSpace:
        def __init__(self, shape):
            self.shape = [shape]
            self.high = np.ones(shape)

    def __init__(self, env, ReplayBuffer, net, act_shape,  delay=80, start_steps=200, update_every=100, iters=None, update_after=0,
                 repl_size=5000, pi_lr=0.001, q_lr=0.001, batch_size=32, gamma=0.99, polyak=0.995, seed=0):
        """

        :param env:
        :param ReplayBuffer:
        :param net:
        :param act_shape: shape of action because it's not env action
        :param delay: wait delay steps till lower agent study, then put obs to buffer after each learn
        :param start_steps: wait before start update - till enough example
        :param update_every: learn agent each update_every steps
        :param iters:
        :param update_after:
        :param repl_size:
        :param pi_lr:
        :param q_lr:
        :param batch_size:
        :param gamma:
        :param polyak:
        """

        super(MetaAgent, self).__init__(env, ReplayBuffer, net, start_steps=200, update_every=100, iters=None, update_after=1000,
                 repl_size=5000, pi_lr=0.001, q_lr=0.001, batch_size=32, gamma=0.99, polyak=0.995, seed=0)
        obs_dim = env.observation_space.shape[0]
        # act_dim=envf.action_space.shape[0]
        self.act_shape = act_shape
        self.action_space = self.ActionSpace(act_shape[-1])
        #print(self.act)
        self._init_net(net, pi_lr, q_lr, self.env.observation_space, self.action_sp)
        self.delay = delay
        self.t_delay = 0



    def store(self, obs, act, rew, next_obs, done):
        self.t += 1
        self.t_delay += 1
        self.log.rput(rew, done)
        if self.t_delay > self.delay:
            # print('store', self.buffer.ptr, self.delay, self.t)
            self.buffer.store(obs, act, rew, next_obs, done)
            if (self.t > self.update_after) and (not self.buffer.ptr % self.update_every):
                #print('train', self.buffer.ptr, end='-')
                self.train()
                self.t_delay = 0

    def policy_correction(self, low_agent, low_buffer):
        for ptr in range(self.buffer.ptr):
            start = self.buffer.obs_buf[ptr]
            goal = self.buffer.act_buf[ptr]
            finish = self.buffer.obs2_buf[ptr]
            # for act in # from start to finish in low_buffer
            #  argmin (a - low_agent.act( # start with goal[1,2,3,4,5,...10] till finish with this goal




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

    ma = MetaAgent(envf, ReplayBuffer, net=MLPActorCritic, act_shape=[envf.observation_space.shape[0]//3])
    obs = envf.reset()

    e = a.get_action(obs)
    print(ma.get_action(obs))

