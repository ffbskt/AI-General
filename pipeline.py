from Sampling_operators import RandomSampling
from models import Model, Trainer
from  env_test import Env
import matplotlib.pyplot as plt
from random import shuffle
from collections import deque

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

model = Model()
env = Env()
t = Trainer(env, model, batch_size=30)
args = dotdict({'cpuct':2, 'iters':100})



def take_best(nodes):

    #print(len(nodes) // 4, len(nodes), nodes[-1].fin_reward)
    return sorted(nodes, key=lambda x: (max(x.fin_reward)), reverse=True)[:len(nodes) // 24]

examples = deque([], maxlen=1000)

for i in range(10):
        #m = MCTS_best_leaf(env, model, args)
        #val = list(m.sampling())
        #print(np.sum([m.Nodes[f].fin_reward > 0 for f in m.Nodes]), [(f.formula, f.predR) for f in get_batch(val)])
    rsmp = RandomSampling(env, model, args)
    rand_val = list(rsmp.sampling())

    #print(sorted([max(x.fin_reward) for x in rand_val]))
    #rand_val = take_best(rand_val)
    #print([max(x.fin_reward) for x in rand_val])
    print('dd', rand_val[0].fin_reward, len(rand_val))
    shuffle(rand_val)

    examples += rand_val
    t.train_model(examples, model)
    print(rsmp.sum_reward)



    # print(policy_net.loss_backet)
plt.plot(t.loss_backet)
plt.show()