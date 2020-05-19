import gym
from replay_buffers import ReplayBuffer, TransisionWrapper
from nets import MLPActorCritic
from games import BitFlipping2, Env_add_Meta
from utils import MiniLog
from agents import DDPGAgent, MetaAgent

size=4
env = BitFlipping2(size, 1, False)
envf = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
a = DDPGAgent(envf, ReplayBuffer, net=MLPActorCritic, start_steps=2000, update_every=50,
              repl_size=10000)
a.buffer = TransisionWrapper(ado=[size,] * 3,obs_dim=envf.observation_space.shape[0], act_dim=envf.action_space.shape[0], size=10000)


ma = MetaAgent(envf, ReplayBuffer, net=MLPActorCritic, act_shape=[envf.observation_space.shape[0] // 3],
               delay=0, start_steps=2000, update_every=50)

envm = Env_add_Meta(envf, [size] * 3, meta_agent=ma, goal_freq=2)


def expirement(steps, agent, env):
     o = env.reset()
     for i in range(steps):
          act = agent.get_action(o)
          o2, r, d, _ = env.step(act)

          agent.store(o, act, r, o2, d)
          o = o2
          if d:
              o = env.reset()


def test_agent(agent, size, rad):
    env = BitFlipping2(size, rad, False)
    test_env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
    log = MiniLog(10)
    for j in range(1000):
        o, d, ep_len = test_env.reset(), False, 0
        while not (d or (ep_len == 2 ** size+4)):
            # Take deterministic actions at test time (noise_scale=0)
            o, r, d, _ = test_env.step(agent.get_action(o, 0))
            log.rput(r, d) # TODO bad test for env with unlimit steps


expirement(10000, a, envm)
#expirement(10000, a, envf)
a.log.rplot()
ma.log.rplot()

ma.log.lplot()
envm.log.rplot()