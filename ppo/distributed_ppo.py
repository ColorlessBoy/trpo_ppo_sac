import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence
import torch.distributed as dist
from ppo import PPO
from time import time
import numpy as np

class DistributedPPO(PPO):
    def __init__(self, 
                actor, 
                critic, 
                clip=0.2, 
                gamma=0.995,
                tau=0.99,
                pi_steps_per_update=80, 
                value_steps_per_update=80,
                target_kl=0.01,
                device=torch.device("cpu"),
                pi_lr=3e-4,
                v_lr=1e-3,
                ent_coef=0.02,
                policy_coef=10.0):
        super(DistributedPPO, self).__init__(actor, critic, clip, gamma, 
                                        tau, pi_steps_per_update, 
                                        value_steps_per_update, 
                                        target_kl, device, pi_lr, v_lr, ent_coef, policy_coef)
        self.synchronous_parameters(self.actor)
        self.synchronous_parameters(self.critic)
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
    
    def synchronous_parameters(self, model):
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
    
    def update_actor(self, state, action, advantage):
        #update actor network
        with torch.no_grad():
            old_pi = self.actor.get_detach_pi(state)
            old_log_action_probs, _ = self.actor.get_log_prob(state, action)

        actor_loss = 0.0
        
        rank = dist.get_rank()
        for i in range(self.pi_steps_per_update):
            self.actor_optim.zero_grad()

            log_action_probs, e = self.actor.get_log_prob(state, action)
            ratio = torch.exp(log_action_probs - old_log_action_probs)
            ratio2 = ratio.clamp(1 - self.clip, 1 + self.clip)
            actor_loss = - self.ent_coef * e.mean() - self.policy_coef * torch.min(ratio * advantage, ratio2 * advantage).mean()
            actor_loss.backward()

            self.average_parameters_grad(self.actor)
            self.actor_optim.step()

            pi = self.actor.get_detach_pi(state)
            kl = kl_divergence(old_pi, pi).sum(axis=1).mean()

            self.average_variables(kl)

            if kl > self.target_kl:
                print("Upto target_kl at Step {}".format(i))
                break

        return actor_loss
    
    def update_critic(self, state, target_value):
        # update critic network
        critic_loss = 0.0
        for _ in range(self.value_steps_per_update):
            value = self.critic(state)
            critic_loss = F.mse_loss(value, target_value)
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.average_parameters_grad(self.critic)
            self.critic_optim.step()
        return critic_loss