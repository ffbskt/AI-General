import numpy as np
import torch
from torch import Tensor, nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnv1 = nn.Conv1d(1,6,8)
        self.fc1 = nn.Linear(42, 182)
        self.fc2 = nn.Linear(182, 40)
        self.action_prob_out = nn.Linear(40, 8)
        #self.val0 = nn.Linear(40, 80)
        self.val = nn.Linear(40, 1)

        #self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.1, weight_decay=0.1)
        #self.loss_backet = []

    def forward(self, x):
        #print(x.shape)
        #x = x.view(-1,17)
        #x = self.cnv1(x.view(1, 1,-1))
        # print(x.shape)
        #x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        act_prob = F.softmax(self.action_prob_out(x), dim=-1)
        val = self.val(x)
        #val_sum = self.val_sum_out(val)

        return act_prob, val

    def predict(self, x):
        self.eval()
        #x = Variable(Tensor(x))
        act_prob, val = self.forward(x)
        return act_prob.data.numpy(), val.data.numpy()
    
    def get_observation(self, formula, env, time):
        #print(formula)
        return Variable(Tensor(env.get_observation(formula, time)))




class Trainer:
    def __init__(self, env, batch_size=10):
        self.env = env
        self.batch_size = batch_size
        self.loss_backet = []

    def clean_unpredict_node(self, nodes_buc):
        new_buck = []
        for node in nodes_buc:
            if len(node.history_data['next_node_ind']):
                new_buck.append(node)
        return new_buck

    def get_batch(self, nodes_buc, batch_size=0):
        """remove all nodes that have not history"""
        n = len(nodes_buc)
        index = [
            np.random.randint(0, n - 1)
            for _ in range(batch_size or self.batch_size)
        ]
        #print(len(self.replay), index, [self.replay[i] for i in index]   )
        return [nodes_buc[i] for i in index]

    def transform_bach_as_input(self, batch, model):
        X = []
        real_prob = []
        real_reward = []
        cur_result = []
        #print(batch)

        for node in batch:
            #if not len(node.history_data['time']):
                
            i = len(node.history_data['time']) -1 #np.random.randint(0, len(node.history_data['time']))
            #if node.parent:
            #self.env.calc_formula(node.parent.formula)


            net_observ = model.get_observation(node.formula, self.env, node.history_data['time'][i])
            X.append(net_observ.view([1,-1]))
            ##p_next = np.zeros(self.env.n_actions)
            ##p_next[node.history_data['next_node_ind'][i]] = 1
            real_prob.append(node.fin_prob) # matrix size(8, 1) of next (f+a) prob
            
            self.env.calc_formula(node.formula)
            cur_result.append(list(self.env.result.values()))

            # real value from past
            if node.parent:
                t = node.history_data['time'][i]
                parent_i = node.parent.history_data['time'].index(t-1)
                real_reward.append(node.parent.history_data['next_node_val'][parent_i])
                #print('ss', X)
            else:
                real_reward.append(node.history_data['next_node_val'][i])
        #print(X)
        X = torch.cat(X, dim=0)
        #print(X.shape)
        real_prob = np.vstack(real_prob)
        real_reward = np.vstack(real_reward)
        cur_result = np.vstack(cur_result)
        return X, real_reward, real_prob, cur_result




    def train_model(self, nodes_buc, model, batch_size=10, net_iters=200):
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.3, weight_decay=0.1)

        for i in range(net_iters):
            batch = self.get_batch(nodes_buc, batch_size=batch_size or self.batch_size)
            # print(batch)

            X, real_reward, real_prob, cr = self.transform_bach_as_input(batch, model)


            #print(i, real_prob.shape)
            model.train()
            #for x, rr, rp in zip(X, real_reward, real_prob):
                #print(xx, rrr,rpp)
            self.optimizer.zero_grad()
            #print(X)
            p_pred, v_pred = model(X)
            # print('pr  ', probability, 'pp  ', p_pred)
            val_loss = torch.mean((Variable(Tensor(real_reward)) - v_pred) ** 2)  # , Variable(Tensor([10]))
            #loss = val_loss - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            loss = - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            #print(loss, rpp, p_pred)
            loss.backward()
            self.optimizer.step()
            self.loss_backet.append(loss.data.numpy())




if __name__ == "__main__":
    from env_test import Env
    from mcts import MCTS

    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

    model = Model()
    #model = LSTMModel(4, 32, 8, 8)
    env = Env()

    args = dotdict({'cpuct':0.5, 'iters':1000})
    mcts = MCTS(env, model, args)
    val = list(mcts.sampling())
    print(len(val))
    val = [v for v in val if v.parent and v.parent.times_visited > 2]
    print(len(val))


    #t = mctsTrainer(env, mcts, batch_size=50)

    t = Trainer(env, batch_size=20)
    for i in range(10):
        print(i, t.loss_backet[-3:])
        mcts = MCTS(env, model, args)
        val = list(mcts.sampling())
        val = t.clean_unpredict_node(val)
        #print('aa', val)
        t.train_model(val, model, net_iters=300)
    # #examples = deque([], maxlen=1000)
    #
    #batch = t.get_batch(val)
    #print(list(t.transform_bach_as_input(batch)[1]))

    import matplotlib.pyplot as plt
    #
    plt.plot(t.loss_backet)
    plt.show()
