from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class MiniLog(object):
    def __init__(self, step=500):
        self.epr = 0
        self.s_epr = []
        self.reward = []
        self.n_interaction = []
        self.t = 0
        self.loss_back = defaultdict(list)
        self.step = step
        self.ep_loss = defaultdict(list)
        self.mean_buff = []
        self.mean_buff_t = []

    def rput(self, r, d):
        self.t += 1
        self.epr += r
        if d:
            self.s_epr.append(self.epr)
            self.epr = 0
        if not self.t % self.step and self.s_epr:
            self.reward.append(np.mean(self.s_epr))
            self.s_epr = []
            self.n_interaction.append(self.t)

            for name in self.ep_loss:
                self.loss_back[name].append(np.mean(self.ep_loss[name]))
                self.ep_loss[name] = []

    def lput(self, loss, name=None):
        if name is None:
            name = 'l'
        self.ep_loss[name].append(loss.data.numpy())
        # if not self.t % self.step:

    def bput(self, mean_buff_rew):
        self.mean_buff.append(mean_buff_rew)
        self.mean_buff_t.append(self.t)

    def rplot(self):
        plt.plot(self.n_interaction, self.reward)
        plt.show()

    def lplot(self):
        for k in self.loss_back:
            plt.plot(self.loss_back[k], label=k)
            plt.legend()
        plt.show()

    def bplot(self):
        plt.plot(self.mean_buff_t, self.mean_buff)

    def pd_transform(self, x='TotalEnvInteracts', y='AverageEpRet', name='log'):
        return pd.DataFrame({x: self.n_interaction, y: self.reward, 'Condition2': name})



def build_agent(env, cls):#, hidden_sizes=(256,256), activation=torch.nn.ReLU):
    #print(env.action_space.shape == ())
    return cls(env.observation_space, env.action_space)
