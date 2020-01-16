import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.distributions import Categorical, Normal
import torch.nn.functional as F
from collections import OrderedDict
import math

def _weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64), activation=torch.relu, output_activation=None):
        super().__init__()
        self.activation = activation
        self.output_activation = output_activation
        self.layers = nn.ModuleList()
        pre_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(pre_size, size))
            pre_size = size
        self.output_layer = nn.Linear(pre_size, output_size)
        self.apply(_weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

# We require that action is in [-1, 1]^n
class PolicyNetwork(Network):
    def __init__(self, input_size, output_size, hidden_sizes=(64, 64),
                 activation=torch.relu, output_activation=torch.tanh, 
                 init_std=1.0, max_log_std=2, min_log_std=-10, epsilon=1e-6):

        super(PolicyNetwork, self).__init__(input_size, hidden_sizes[-1], hidden_sizes[-1:], activation, activation)

        self.log_std_bias = 0.5 * (max_log_std + min_log_std)
        self.log_std_scale = 0.5 * (max_log_std - min_log_std)
        self.output_activation = output_activation
        self.epsilon = epsilon

        self.mean_layers = nn.Linear(hidden_sizes[-1], output_size)
        self.std_layers = nn.Linear(hidden_sizes[-1], output_size)
        self.apply(_weight_init)

    def forward(self, x):
        x = super(PolicyNetwork, self).forward(x)
        mean = self.mean_layers(x)
        log_std = self.std_layers(x)
        if self.output_activation:
            mean = self.output_activation(mean)
        # Spinningup method.
        log_std = torch.tanh(log_std)
        log_std = self.log_std_bias + self.log_std_scale * log_std
        return Normal(loc=mean, scale=log_std.exp())
    
    def select_action(self, state):
        pi = self.forward(state)
        y = pi.rsample()
        # y = pi.sample()
        action = torch.tanh(y)
        log_pi_action = pi.log_prob(y) - torch.log(1 - action.pow(2) + self.epsilon)
        return action, log_pi_action.sum(axis=1, keepdim=True)

    def get_mean_action(self, state):
        pi = self.forward(state)
        y = pi.loc
        action = torch.tanh(y)
        return action

class ValueNetwork(Network):
    def __init__(self, input_size, hidden_sizes=(64, 64), \
                activation=torch.relu, output_activation=None):
        super(ValueNetwork, self).__init__(input_size, 1, hidden_sizes,
                                            activation, output_activation)

class QNetwork(Network):
    def __init__(self, state_size, action_size, hidden_sizes=(64,64), \
                activation=torch.relu, output_activation=None):
        super(QNetwork, self).__init__(state_size + action_size, 1, hidden_sizes, 
                                        activation, output_activation)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        return super(QNetwork, self).forward(x)

if __name__ == '__main__':
    state_size = 10
    action_size = 4
    policy = PolicyNetwork(state_size, action_size, (64, 64))
    v_net = ValueNetwork(state_size)
    q_net = QNetwork(state_size, action_size)
    state = torch.randn((2, 10))
    action, _ = policy.select_action(state)
    print("action = ", action)

    print(v_net(state))

    print(q_net(state, action))

    print(policy)

