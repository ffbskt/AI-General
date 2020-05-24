import gym
from replay_buffers import ReplayBuffer, TransisionWrapper
from nets import MLPActorCritic
from games import BitFlipping2, Env_add_Meta
from utils import MiniLog
from agents import DDPGAgent, HIRO


size=5
env = BitFlipping2(size, 1, False)
envf = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
a = DDPGAgent(envf, ReplayBuffer, net=MLPActorCritic, start_steps=3000, update_every=50,
              repl_size=10000)
a.buffer = TransisionWrapper(ado=[size,] * 3,obs_dim=envf.observation_space.shape[0], act_dim=envf.action_space.shape[0], size=10000)

agent_kwargs = dict(env=envf, replay_buffer=ReplayBuffer, net=MLPActorCritic, start_steps=3000, update_every=50, repl_size=10000)
hiro_kwargs = dict(env=envf, net=MLPActorCritic, act_shape=[envf.observation_space.shape[0] // 3], start_steps=200, update_every=50)
#print(hiro_kwargs.update(agent_kwargs))
hiro = HIRO(envf, low_agent_kwargs=agent_kwargs, high_agent_kwargs=agent_kwargs)#, net=MLPActorCritic, act_shape=[envf.observation_space.shape[0] // 3],
#               start_steps=200, update_every=50)
#hiro.low_agent.buffer = TransisionWrapper(ado=[size,] * 3,obs_dim=envf.observation_space.shape[0], act_dim=envf.action_space.shape[0], size=10000)
#envm = Env_add_Meta(envf, [size] * 3, meta_agent=ma, goal_freq=2)


def expirement(steps, agent, env, seed=(0, 3), agent_name='agent', model_args={}):
    o = env.reset()
    log = MiniLog(100, kwargs=model_args)
    for s in seed:
        agent.reset_agent(s)
        for i in range(steps):
            act = agent.get_action(o)
            o2, r, d, _ = env.step(act)
            log.rput(r, d)
            agent.store(o, act, r, o2, d)
            o = o2
            if d:
              o = env.reset()
        log.pd_append(name=agent_name)
    log.save()
    return log

env = BitFlipping2(size, 1, False)
test_env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
def test_agent(test_env, agent):
    sum_r = 0
    n = 100
    for j in range(n):
        o, d, ep_len = test_env.reset(), False, 0
        while not (d or (ep_len == 2 ** 7+4)):
            # Take deterministic actions at test time (noise_scale=0)
            o, r, d, _ = test_env.step(agent.get_action(o, 0))
            sum_r += r # TODO bad test for env with unlimit steps
    return sum_r / n


log = expirement(3000, hiro, envf, model_args={**hiro_kwargs, **agent_kwargs})
#expirement(10000, a, envf)
#print(log.pd_data.to_csv('data_log/hiro0_plus_her.csv'))
log.rplot()

#ma.log.rplot()

#ma.log.lplot()

from spinup import ddpg_pytorch
import spinup.algos.pytorch.ddpg.core as core
import argparse

def make_env(size=7):
    env = BitFlipping2(size, 1, False)
    envf = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
    return envf

parser = argparse.ArgumentParser()
#parser.add_argument('--env', type=str, default='HalfCheetah-v2')
parser.add_argument('--hid', type=int, default=256)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--exp_name', type=str, default='ddpgbfl')
parser.add_argument('--start_steps', type=int, default=2000)
parser.add_argument('--steps_per_epoch', type=int, default=300)
args = parser.parse_args()

from spinup.utils.run_utils import setup_logger_kwargs
logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#ddpg_pytorch(make_env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
#     gamma=args.gamma, seed=args.seed, epochs=args.epochs, start_steps=2000, update_after=300, steps_per_epoch=500, logger_kwargs=logger_kwargs)