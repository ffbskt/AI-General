import gym
from replay_buffers import ReplayBuffer, TransisionWrapper
from nets import MLPActorCritic
from games import BitFlipping2, Env_add_Meta
from utils import MiniLog
from agents import DDPGAgent, HIRO

project_dir = ''
import gym
from replay_buffers import ReplayBuffer, TransisionWrapper
from nets import MLPActorCritic
from games import BitFlipping2, Env_add_Meta
from utils import MiniLog
from agents import DDPGAgent, HIRO
# Default options
size=3
env_kw = dict(size=size, rad=1, discret_space=False, seed=0)
# env = BitFlipping2(**env_kw)
# envf = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))

envf = gym.make('LunarLanderContinuous-v2')
print(envf.observation_space, envf.action_space, envf.reset())


agent_kwargs = dict(env=envf, replay_buffer=ReplayBuffer, net=MLPActorCritic, start_steps=200, update_every=100, iters=None, update_after=100,
                 repl_size=10000, pi_lr=0.001, q_lr=0.001, batch_size=32, gamma=0.99, polyak=0.995,
                 seed=0, act_noise=0.1, act_limit=None)
hagent_kwargs = agent_kwargs


def expirement(steps, agent, env, seed=(0, 2, 3), agent_name='agent', model_args={}):
    o = env.reset()
    log = MiniLog(50, kwargs=model_args, save_dir=project_dir + 'data_log',
                  all_log_navigate=project_dir + 'data_log/log_navi')
    print(log.time, end='.csv ')
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




### pre test
a = DDPGAgent(envf, ReplayBuffer, net=MLPActorCritic, start_steps=3000, update_every=50, repl_size=10000)
o = envf.reset()
expirement(10000, a, envf, model_args={**agent_kwargs, **hagent_kwargs})

def test_agent(agent):
    sum_r = 0
    n = 100
    env = BitFlipping2(size, 1, False)
    test_env = gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(env))
    for j in range(n):
        o, d, ep_len = test_env.reset(), False, 0
        while not (d or (ep_len == 2 ** 7 + 4)):
            # Take deterministic actions at test time (noise_scale=0)
            o, r, d, _ = test_env.step(agent.get_action(o, 0))
            sum_r += r  # TODO bad test for env with unlimit steps
    return sum_r / n


hiro_kwargs = dict(store_delay=20, train_delay=10, step_each=1, ado=None)

#hiro = HIRO(envf, low_agent_kwargs=agent_kwargs, high_agent_kwargs=hagent_kwargs, **hiro_kwargs)

#for p in [0.1, 0]:
#  hagent_kwargs['act_noise'] = p
#  hiro = HIRO(envf, low_agent_kwargs=agent_kwargs, high_agent_kwargs=hagent_kwargs, **hiro_kwargs)
#  expirement(1000, hiro, envf, model_args={**agent_kwargs, **hagent_kwargs})
#  print(hiro.high_agent.act_noise, test_agent(hiro))
#for p in [1, 10]:
#  hagent_kwargs['act_limit'] = p
#  hiro = HIRO(envf, low_agent_kwargs=agent_kwargs, high_agent_kwargs=hagent_kwargs, **hiro_kwargs)
#  expirement(1000, hiro, envf, model_args={**agent_kwargs, **hagent_kwargs})
#  print(hiro.high_agent.act_noise, test_agent(hiro))
