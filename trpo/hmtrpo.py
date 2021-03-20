import torch
import torch.distributed as dist
import torch.nn.functional as F
from local_trpo import LocalTRPO

from time import time

class HMTRPO(LocalTRPO):
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
        super(HMTRPO, self).__init__(actor, critic, value_lr,
                                    value_steps_per_update,
                                    cg_steps, linesearch_steps,
                                    gamma, tau, damping, max_kl, device)

    def average_parameters_grad(self, model):
        size = float(dist.get_world_size())
        rank = dist.get_rank()
        for param in model.parameters():
            dist.reduce(param.grad.data, dst=0)
            if rank == 0:
                param.grad.data /= size

    def get_actor_loss_grad(self, state, action, advantage):
        loss_grad = super(HMTRPO, self).get_actor_loss_grad(state, action, advantage)
        # Average actor_loss_grad.
        self.average_variables(loss_grad)
        return loss_grad
    
    def cg(self, A, b, iters=10, accuracy=1e-10):
        x = super(HMTRPO, self).cg(A, b, iters, accuracy)
        self.average_variables(x)
        return x

    def linesearch(self, state, action, advantage, fullstep, steps=10):
        start_time = time()
        self.average_variables(fullstep)
        actor_loss = super(HMTRPO, self).linesearch(state, action, advantage, fullstep, steps)
        print('HMTRPO linesearch() uses {}s.'.format(time() - start_time)) 
        return actor_loss

    def update_critic(self, state, target_value):
        start_time = time()
        rank = dist.get_rank()
        critic_loss = 0.0
        for _ in range(self.value_steps_per_update):
            value = self.critic(state)
            critic_loss = F.mse_loss(value, target_value)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.average_parameters_grad(self.critic)
            if rank == 0:
                self.critic_optim.step()
            self.synchronous_parameters(self.critic)
        print("GlobalTRPO updates critic by using {}s".format(time() - start_time))
        return critic_loss