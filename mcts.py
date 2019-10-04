import numpy as np
from env_test import Env
from policy import Model
from collections import defaultdict
import heapq

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

class Node:
    def __init__(self, formula, predP, predR, fin_reward=0):
        self.formula = formula
        self.predP = predP
        self.predR = predR
        self.fin_reward = fin_reward
        self.N = 0
        self.Q = 0

    def __lt__(self, other):
        if self.fin_reward:
            return False
        if other.fin_reward:
            return True
        return self.predR > other.predR

class MCTS_best_leaf:
    """
        different algorithm. The Idea do not go down by tree choose best ucb node, except - we immediatly
        compare:
     1) leafs and go to best leaf even parents branch has low ucb.
     2) compute_Q - no value 

    """
    def __init__(self, env, model, args):
        self.env = env
        self.args = args
        node = Node('', 1, 0.5)
        self.heap_best = [node,]
        self.Nodes = {'': node}
        #self.Ns[node.formula]
        self.model = model
        #self.samples = []

    def find_best_leaf(self):
        return heapq.heappop(self.heap_best)

    def expand(self, node):
        #print(node.fin_reward)
        pred_P, pred_R = self.model.predict(self.env.get_observation(node.formula))
        self.up_prev_N(node)
        for a in range(self.env.n_actions):
            r, next_formula = self.env.do_move(node.formula, a)
            Q = self.compute_Q(pred_P[a], pred_R[a], next_formula)
            new_node = Node(next_formula, pred_P[a], pred_R[a], fin_reward=r)
            #print(next_formula, r)
            heapq.heappush(self.heap_best, new_node) ###&&&
            if r != 0:
                self.Nodes[next_formula] = new_node

        return r

    def up_prev_N(self, node):
        #print(self.Ns)
        formula = node.formula
        if formula in self.Nodes:
            return 0
        self.Nodes[formula] = node
        self.Nodes[formula].N += 1
        for i in range(len(formula)):
            #print(formula, formula[:-i - 1], i)
            self.Nodes[formula[:-i]].N += 1

    def compute_Q(self, pred_P, pred_R, formula):
        # print(formula, pred_R,  self.args.cpuct * pred_P * np.sqrt(self.Ns[formula[:-1]]), self.Ns[formula[:-1]], self.Ns, formula[:-1])
        return pred_R + self.args.cpuct * pred_P * np.sqrt(self.Nodes[formula[:-1]].N)

    def sampling(self):
        i, r = 0, 0
        while (args.iters > i and not r) or 2 * args.iters > i:
            node = m.find_best_leaf()
            r = max(m.expand(node), r)
            i += 1

        # TODO add_real_reward(self.Nodes)
        self.add_real_reward()

        return self.Nodes.values()


    def add_real_reward(self):
        sorted_formulas = sorted(list(m.Nodes), key=lambda x: (m.Nodes[x].fin_reward, len(x)))
        positive_ind = 1
        for i, f in enumerate(sorted_formulas):
            if self.Nodes[f].fin_reward > 0:
                positive_ind = i
            self.Nodes[f].fin_reward *= np.sqrt(positive_ind/(1 + i))




if __name__ == "__main__":
    model = Model()
    env = Env(inp=1, out=1)

    args = dotdict({'cpuct':2, 'iters':22})
    m = MCTS_best_leaf(env, model, args)





    #m.add_real_reward()
    #print(list(m.Nodes))
    #print(sorted(list(m.Nodes), key=lambda x: (m.Nodes[x].fin_reward, len(x))))
    #print([(k, len(k), n.fin_reward) for k, n in m.Nodes.items()])
    # TODO net_train
    #for n in m.sampling().items():
    #    print(n)
    from policy import NN_input

    def prepare_input(env, inp, out, formula):
        #env.formula = formula
        env.inp, env.out = inp, out
        env.calc_formula(formula)
        values, err = env.result, env.err
        net_observ = NN_input(inp, out, formula, values, err)
        return net_observ


    def get_batch(nodes_buc, batch_size=10):
        n = len(nodes_buc)
        index = [
            np.random.randint(0, n - 1)
            for _ in range(batch_size)
        ]
        #print(len(self.replay), index, [self.replay[i] for i in index]   )
        return [nodes_buc[i] for i in index]





    def train_model(nodes_buc):
        for i in range(20):
            batch = get_batch(nodes_buc, batch_size=10)
            # print(batch)
            X = np.vstack([prepare_input(env, env.inp, env.out, node.formula)
                           for node in batch])
            reward = np.vstack([np.array([node.fin_reward]) for node in batch])
            probability = np.vstack([np.array([node.predP]) for node in batch])
            print('s2', model)
            model.training(X, reward, probability)
            print('s', model)


    val = list(m.sampling())

    train_model(val)















































#    import matplotlib.pyplot as plt
    #print(policy_net.loss_backet)    
#    plt.plot(policy_net.loss_backet)
#    plt.show()
    


# while root.children:
    #print(root.children.pop().children.pop().env.formula)
#    root = root.select_best_leaf()
#    print('ff', root.env.formula)
