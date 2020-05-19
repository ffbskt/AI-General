import numpy as np
class Node:
    parent = None  # parent Node
    value_sum = 0.  # sum of state values from all visits (numerator)
    times_visited = 0
    def __init__(self, parent=None, action=None, prob=None, env=None):
        self.parent = parent
        self.action = action
        self.P = prob        # if both prob needed ??
        self.fin_prob = 0    # this prob filled after sampling (times choose !this! node from other)
        self.temp_prob = []  # this prob filled during sampling (algorithm prob of chose !next! node)
        self.history_data = {'time':[], 'next_node_ind':[], 'next_node_val':[]}
        self.children = set()
        if parent:
            self.immediate_reward, self.formula = env.do_move(parent.formula, action)
            #if self.immediate_reward > 0: print("!!-----------------------------", self.formula)
        else:
            self.immediate_reward, self.formula = 0.0, ''



    def is_leaf(self):
        return len(self.children) == 0

    def get_mean_value(self):
        return self.value_sum / self.times_visited if self.times_visited != 0 else 0

    def ucb_score(self, scale=10, max_value=1e5):  # 1e100
        #print('ucb ', self.P, self.parent.times_visited)
        U = (self.P *
             np.sqrt(self.parent.times_visited) / (1 + self.times_visited))  # need if zero visited?
        return self.get_mean_value() + scale * U


class MCTS:
    def __init__(self, env, model, args):
        """

        :param env:
        :param model:
        :param args:
        """
        self.env = env
        self.model = model
        self.args = args
        node = Node()
        self.Nodes = {'': node}
        self.sum_reward = 0
        self.root = Node()
        self.iter_timer = 0


    def get_action_prob(self, temp=0):
        # TODO choose action each node?
        node = self.root
        #print('node0', node.formula)
        i = 0
        while not node.is_leaf(): #and not self.env.game_end(node.formula):
            #print(node.formula, self.root.children)
            i += 1
            #if node.is_leaf():
            #    action_prob = [1 / self.env.n_actions] * self.env.n_actions
            #    print('node is leaf', node.formula)
            #    return action_prob, node

            self.iter_timer += 1
            node.history_data['time'].append(self.iter_timer)
            next_node = max(node.children, key=lambda x: x.ucb_score())
            #print([[i.ucb_score(), i.formula] for i in node.children], next_node.formula)
            choosen_act = next_node.formula[-1]
            node.history_data['next_node_ind'].append(self.env.action_space.index(choosen_act))
            node.history_data['next_node_val'].append(next_node.immediate_reward)
            #print(node.formula, len(node.children), next_node.formula)
            node = next_node
            #print('ret', node.formula)

            if next_node.is_leaf():

                #print('next_node is leaf', node.formula)
                if temp == 0:
                    action_prob = [0] * self.env.n_actions
                    action_index = self.env.action_space.index(choosen_act)
                    action_prob[action_index] = 1

                    return action_prob, node#.parent

                ucb = [child.ucb_score() ** (1. / temp) for child in node.parent.children]
                ucb_sum = sum(ucb)
                action_prob = [x / ucb_sum for x in ucb]
                return action_prob, node.parent

        action_prob = [1 / self.env.n_actions] * self.env.n_actions
        return action_prob, node


            #print(i, 'sss', node.formula)
            #print(next_node.is_leaf())

    def select_best_leaf(self, temp=0.1):
        prob, parent_node = self.get_action_prob(temp)
        #print('f, p', parent_node.formula, prob)
        parent_node.temp_prob = prob
        if parent_node.children:
            node = np.random.choice(list(parent_node.children), p=prob)

            return node
        return parent_node


    def expand(self, node, action_priors):
        #print('expand ', node.formula)
        for action, prob in enumerate(action_priors):
            next_node = Node(node, action, prob, self.env)
            node.children.add(next_node)
            self.sum_reward += max(0, next_node.immediate_reward)

    def propagate(self, node, fogot=0.9):
        node.value_sum += node.immediate_reward
        R = node.immediate_reward
        node.times_visited += 1
        while node.parent:
            node = node.parent
            node.value_sum += R #child_val
            R *= fogot
            node.times_visited += 1

    def sampling(self, temp=0):
        i, r = 0, 0
        while (self.args.iters > i and not r) or 2 * self.args.iters > i:

            node = self.select_best_leaf(temp)
            r = node.immediate_reward
            self.Nodes[node.formula] = node
            if r:
                self.propagate(node)
            else:

                pred_P, pred_R = self.model.predict(self.model.get_observation(node.formula, self.env, self.iter_timer)) # if simple model self.env.observation
                self.expand(node, pred_P)

            r = max(0, r)
            i += 1

        #self.compute_p_real()
        return  self.Nodes.values()


if __name__ == "__main__":
    # from pipeline import dotdict
    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    import models
    from env_test import Env

    model = models.Model()
    args = dotdict({'cpuct': 1, 'iters': 400})
    env = Env(inp=1, out=1)
    rsmp = MCTS(env, model, args)
    rand_val = []
    for i in range(1):
        rsmp.env = Env(inp=1, out=i+1)
        print('out ', i+1)
        rand_val += list(rsmp.sampling(temp=0))

    #print([f.formula for f in rand_val])
    print('T', rsmp.iter_timer, 'ss')

    print([(f.formula, f.ucb_score(), f.times_visited) for f in rand_val if f.formula[:2]=='ie' and len(f.formula)<14])
    print([(f.formula, f.fin_prob, f.ucb_score(), f.get_mean_value(), f.times_visited) for f in rand_val if
           f.formula[:1] == 'i' and len(f.formula) < 3])
    print(rsmp.Nodes[''].times_visited)

    #print(np.random.choice([env, model], p=[0.5, 0.5]))

    import matplotlib.pyplot as plt
    #plt.plot([f.ucb_score() for f in rand_val if f.formula != ''])
    plt.plot(rsmp.Nodes['ie'].history_data['time'])
    plt.show()
