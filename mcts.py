import numpy as np




class Node:
    parent = None  # parent Node
    value_sum = 0.  # sum of state values from all visits (numerator)
    times_visited = 0
    def __init__(self, parent=None, action=None, prob=None, env=None):
        self.parent = parent
        self.action = action
        self.P = prob
        self.children = set()
        if parent:
            self.immediate_reward, self.formula = env.do_move(parent.formula, action)
            # if self.immediate_reward > 0: print("!!-----------------------------", self.formula)
        else:
            self.immediate_reward, self.formula = 0.0, ''

        self.fin_prob = 0

    def is_leaf(self):
        return len(self.children) == 0

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e5):  # 1e100
        """
        Computes ucb1 upper bound using current value and visit counts for node and it's parent.

        :param scale: Multiplies upper bound by that. From hoeffding inequality, assumes reward range to be [0,scale].
        :param max_value: a value that represents infinity (for unvisited nodes)

        """

        U = (self.P *
             np.sqrt(self.parent.times_visited) / (1 + self.times_visited))  # need if zero visited?

        # if self.get_mean_value() > 0:
        #    print(self.get_mean_value(), self.env.formula)

        # print(self.env.formula, U, self.P, self.parent.times_visited)
        return self.get_mean_value() + scale * U


class MCTS:
    def __init__(self, env, model, args):
        self.env = env
        self.model = model
        self.args = args
        node = Node()
        self.Nodes = {'': node}
        self.sum_reward = 0
        self.root = Node()


    def select_best_leaf(self):
        node = self.root
        while not node.is_leaf():
            #print(node.formula, [i.value_sum for i in node.children])
            node = max(node.children, key=lambda x: x.ucb_score())
        return node

    def expand(self, node, action_priors):
        for action, prob in enumerate(action_priors):
            node.children.add(Node(node, action, prob, self.env))

    def propagate(self, node, fogot=0.9):
        #print(1, node.formula,node.immediate_reward, node.value_sum)
        #if node.formula[0] == 'i' and len(node.formula)==2: print(node.parent.value_sum, node.parent.times_visited)
        node.value_sum += node.immediate_reward
        R = node.immediate_reward
        node.times_visited += 1
        while node.parent:
            #if node.formula[0] == 'i' and len(node.formula) == 2: print(node.parent.value_sum,
            #                                                            node.parent.times_visited)
            #child_val = node.value_sum
            node = node.parent
            node.value_sum += R #child_val
            R *= fogot
            node.times_visited += 1

    def sampling(self):
        i, r = 0, 0
        while (self.args.iters > i and not r) or 2 * self.args.iters > i:
            node = self.select_best_leaf()
            #if node.parent is not None: print(node.formula, node.immediate_reward)
            r = node.immediate_reward
            #if r: print(node.formula)
            self.Nodes[node.formula] = node
            if r:
                self.propagate(node)
            else:
                pred_P, pred_R = self.model.predict(self.env.get_observation(node.formula))
                self.expand(node, pred_P)

            r = max(0, r)
            i += 1

        self.compute_p_real()
        return  self.Nodes.values()

    def compute_p_real(self):
        for f in self.Nodes:
            if not self.Nodes[f[:-1]].times_visited: print(self.Nodes[f[:-1]].formula,
                                                           self.Nodes[f].formula, self.Nodes[f].times_visited)
            if self.Nodes[f].times_visited:
                self.Nodes[f].fin_prob = self.Nodes[f].times_visited / self.Nodes[f[:-1]].times_visited

if __name__ == "__main__":
    # from pipeline import dotdict
    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    import models
    from env_test import Env

    model = models.Model()
    args = dotdict({'cpuct': 2, 'iters': 1000})
    env = Env(inp=1, out=1)
    rsmp = MCTS(env, model, args)
    rand_val = []
    for i in range(1):
        rsmp.env = Env(inp=1, out=i+1)
        print(123)
        rand_val += list(rsmp.sampling())

    print([(f.formula, f.immediate_reward, f.ucb_score(), f.times_visited) for f in rand_val if f.formula[:2]=='ie' and len(f.formula)<4])
    print([(f.formula, f.fin_prob, f.ucb_score(), f.get_mean_value(), f.times_visited) for f in rand_val if
           f.formula[:1] == 'i' and len(f.formula) < 3])
    print(rsmp.Nodes[''].times_visited)

    import matplotlib.pyplot as plt
    plt.plot([f.ucb_score() for f in rand_val if f.formula != ''])
    plt.show()

