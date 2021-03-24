import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from local_trpo import LocalTRPO

from time import time

class DistributedTRPO(LocalTRPO):
    def __init__(self, 
                actor, 
                critic, 
                value_lr=0.01,
                value_steps_per_update=50,
                cg_steps=10,
                linesearch_steps=10,
                gamma=0.99,
                tau=0.97,
                damping=0.1,
                max_kl=0.01,
                device=torch.device("cpu")):
        super(DistributedTRPO, self).__init__(actor, critic, value_lr,
                                    value_steps_per_update,
                                    cg_steps, linesearch_steps,
                                    gamma, tau, damping, max_kl, device)

        self.weights=np.array([[0.6, 0.2, 0,   0,    0,      0,    0.2,  0],
                        [0.2, 0.2, 0.2, 0,    0,      0,    0.2,  0.2],
                        [0,   0.2, 0.8, 0,    0,      0,    0,    0],
                        [0,   0,   0,   0.55, 0,      0.25, 0.20, 0],
                        [0,   0,   0,   0,    0.4167, 0.25, 0,    0.3333],
                        [0,   0,   0,   0.25, 0.25,   0.3,  0.2,  0],
                        [0.2, 0.2, 0,   0.2,  0,      0.2,  0.2,  0],
                        [0,   0.2, 0,   0,    0.3333, 0,    0,    0.4667]])
    
    def average_parameters_grad(self, model):
        for param in model.parameters():
            self.average_variables(param)

    def average_variables(self, variables):
        size = dist.get_world_size()
        rank = dist.get_rank()
        tensor_list = [torch.empty_like(variables) for _ in range(size)]
        dist.all_gather(tensor_list, variables)
        variables.data.mul_(self.weights[rank][rank])
        for n, tensor in enumerate(tensor_list):
            if n != rank and self.weights[rank][n] > 0:
                variables.data.add_(self.weights[rank][n] * tensor)

    def get_actor_loss_grad(self, state, action, advantage):
        loss_grad = super(DistributedTRPO, self).get_actor_loss_grad(state, action, advantage)
        # Average actor_loss_grad.
        self.average_variables(loss_grad)
        return loss_grad
    
    def cg(self, A, b, iters=10, accuracy=1e-10):
        x = super(DistributedTRPO, self).cg(A, b, iters, accuracy)
        self.average_variables(x)
        return x

    def linesearch(self, state, action, advantage, fullstep, steps=10):
        self.average_variables(fullstep)
        actor_loss = super(DistributedTRPO, self).linesearch(state, action, advantage, fullstep, steps)
        return actor_loss

    def update_critic(self, state, target_value):
        critic_loss = 0.0
        for _ in range(self.value_steps_per_update):
            value = self.critic(state)
            critic_loss = F.mse_loss(value, target_value)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.average_parameters_grad(self.critic)
            self.critic_optim.step()
        return critic_loss