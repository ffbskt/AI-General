import numpy as np
from env_test import Env
from policy import Model
from best_leaf import Node, dotdict
import bisect


class RandomSample:
    def __init__(self, env, model, args):
        self.env = env
        self.args = args
        node = Node('', 1, 0.5)
        self.Nodes = {'':node}
        self.model = model

    def sampling(self):
        i, r = 0, 0
        while (args.iters > i and not r) or 2 * args.iters > i:
            formula_len = np.random.choice(range(24))
            formula = ''
            for step in range(formula_len):
                pred_P, pred_R = self.model.predict(self.env.get_observation(formula))
                #print(pred_P)
                a = np.random.choice(range(8), p=pred_P)
                r, formula = self.env.do_move(formula, a)
                if formula not in self.Nodes:
                    self.Nodes[formula] = Node(formula, pred_P, pred_R, fin_reward=r, Q=r)
                    if r:
                        break
                else:
                    self.Nodes[formula].N += 1



            r = max(0, r)
            i += 1

            if r > 0:
                print('!!', formula)

        return self.Nodes.values()


if __name__ == "__main__":
    model = Model()
    env = Env(inp=1, out=6)

    args = dotdict({'cpuct':2, 'iters':2000})

    m = RandomSample(env, model, args)
    m.sampling()




    #m.add_real_reward()
    #print(list(m.Nodes))
    #print(sorted(list(m.Nodes), key=lambda x: (m.Nodes[x].fin_reward, len(x))))
    #print([(k, len(k), n.fin_reward) for k, n in m.Nodes.items()])
    # TODO net_train
    #for n in m.sampling().items():
    #    print(n)
    from policy import NN_input