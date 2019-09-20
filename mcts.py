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
        return self.predR < other.predR

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
        self.Ns = defaultdict(int)
        #self.Ns[node.formula]
        self.model = model

    def find_best_leaf(self):
        return heapq.heappop(self.heap_best)

    def expand(self, node):
        pred_P, pred_R = self.model.predict(self.env.get_observation(node.formula))
        self.up_prev_N(node.formula)
        for a in range(self.env.n_actions):
            r, next_formula = self.env.do_move(node.formula, a)
            Q = self.compute_Q(pred_P[a], pred_R[a], next_formula)
            node = Node(next_formula, pred_P[a], pred_R[a])
            print(Q, node)
            heapq.heappush(self.heap_best, node) ###&&&

    def up_prev_N(self, formula):
        for i in range(len(formula)):
            self.Ns[formula[:-i]] += 1

    def compute_Q(self, pred_P, pred_R, formula):
        return pred_R + self.args.cpuct * pred_P * np.sqrt(self.Ns[formula[:-1]])


#if __name__ == "__main__":
model = Model()
env = Env()
args = dotdict({'cpuct':2, 'iters':3})
m = MCTS_best_leaf(env, model, args)

node = m.find_best_leaf()
m.expand(node)

print(m.heap_best)






















































#    import matplotlib.pyplot as plt
    #print(policy_net.loss_backet)    
#    plt.plot(policy_net.loss_backet)
#    plt.show()
    


# while root.children:
    #print(root.children.pop().children.pop().env.formula)
#    root = root.select_best_leaf()
#    print('ff', root.env.formula)
