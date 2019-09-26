import numpy as np
import torch
from torch import Tensor, nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(41, 82)
        self.fc2 = nn.Linear(82, 40)
        self.action_prob_out = nn.Linear(40, 8)
        #self.val0 = nn.Linear(40, 80)
        self.val = nn.Linear(40, 8)

        self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.1, weight_decay=0.1)
        self.loss_backet = []

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        act_prob = F.softmax(self.action_prob_out(x), dim=-1)
        val = F.tanh(self.val(x))
        #val_sum = self.val_sum_out(val)

        return act_prob, val

    def predict(self, x):
        x = Variable(Tensor(x))
        act_prob, val = self.forward(x)
        return act_prob.data.numpy(), val.data.numpy()




class Trainer:
    def __init__(self, env, states=None, batch_size=10):
        self.env = env
        self.batch_size = batch_size
        self.loss_backet = []

    def get_batch(self, nodes_buc, batch_size=0):
        n = len(nodes_buc)
        index = [
            np.random.randint(0, n - 1)
            for _ in range(batch_size or self.batch_size)
        ]
        #print(len(self.replay), index, [self.replay[i] for i in index]   )
        return [nodes_buc[i] for i in index]

    def transform_bach_as_input(self, batch):
        X = []
        real_prob = []
        real_reward = []
        for node in batch:
            self.env.calc_formula(node.formula)
            net_observ = self.env.NN_input(node.formula)
            X.append(net_observ)
            real_prob.append(node.fin_prob) # matrix size(8, 1) of next (f+a) prob
            real_reward.append(node.fin_reward)
        X = np.vstack(X)
        real_prob = np.vstack(real_prob)
        real_reward = np.vstack(real_reward)
        return X, real_reward, real_prob




    def train_model(self, nodes_buc, model):
        self.optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=0.1, weight_decay=0.1)

        for i in range(200):
            batch = self.get_batch(nodes_buc, batch_size=35)
            # print(batch)

            X, real_reward, real_prob = self.transform_bach_as_input(batch)

            self.optimizer.zero_grad()
            p_pred, v_pred = model(Variable(Tensor(X)))
            # print('pr  ', probability, 'pp  ', p_pred)
            val_loss = torch.mean((Variable(Tensor(real_reward)) - v_pred) ** 2)  # , Variable(Tensor([10]))
            loss = val_loss - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            #loss = - torch.mean(Variable(Tensor(real_prob)) * torch.log(p_pred))
            loss.backward()
            self.optimizer.step()
            self.loss_backet.append(loss.data.numpy())