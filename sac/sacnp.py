import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

from utils import soft_update

class SACNP(object):
    def __init__(self, v_net, q1_net, q2_net, pi_net, vt_net, 
                gamma=0.99, alpha=0.2, lm = 1,
                v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = 0.005,
                device=torch.device('cpu')):
        # nets
        self.v_net, self.q1_net, self.q2_net, self.pi_net, self.vt_net = \
            v_net, q1_net, q2_net, pi_net, vt_net

        # hyperparameters
        self.gamma, self.alpha, self.lm = gamma, alpha, lm

        # device
        self.device = device

        # optimization
        self.v_optim  = Adam(self.v_net.parameters(),  lr = v_lr )
        self.q1_optim = Adam(self.q1_net.parameters(), lr = q_lr )
        self.q2_optim = Adam(self.q2_net.parameters(), lr = q_lr)
        self.pi_optim = Adam(self.pi_net.parameters(), lr = pi_lr)
        self.vt_optim = SGD(self.vt_net.parameters(), lr = vt_lr)

    def update(self, batch1, batch2):
        def to_device(batch):
            state  = torch.FloatTensor(batch[0]).to(self.device)
            action = torch.FloatTensor(batch[1]).to(self.device)
            reward = torch.FloatTensor(batch[2]).to(self.device).unsqueeze(1)
            nstate = torch.FloatTensor(batch[3]).to(self.device)
            mask   = torch.FloatTensor(batch[4]).to(self.device).unsqueeze(1)
            return state, action, reward, nstate, mask

        def step(net_list, net_optim_list, loss_grad_list, ht_grad_list):
            for net, net_optim, loss_grad, ht_grad in zip(net_list, net_optim_list, loss_grad_list, ht_grad_list):
                prev_ind = 0
                new_loss_grad = loss_grad + self.lm * ht_grad
                for param in net.parameters():
                    flat_size = param.numel()
                    param.grad = \
                        new_loss_grad[prev_ind:prev_ind + flat_size].view(param.size())
                    prev_ind += flat_size
                net_optim.step()
        
        batch1 = to_device(batch1) 
        batch2 = to_device(batch2)

        net_list = (self.q1_net, self.q2_net, self.pi_net, self.v_net, self.vt_net)
        optim_list = (self.q1_optim, self.q2_optim, self.pi_optim, self.v_optim, self.vt_optim)

        loss_list_1 = self.get_loss_list(*batch1)
        loss_grad_list_1 = self.get_loss_grad_list(loss_list_1, net_list)

        loss_list_2 = self.get_loss_list(*batch2)
        ht_grad_list = self.get_ht_grad_list(loss_list_2, net_list, loss_grad_list_1)

        step(net_list, optim_list, loss_grad_list_1, ht_grad_list)
        return loss_list_1

    def get_loss_list(self, state, action, reward, nstate, mask):
        q_target = reward + self.gamma * mask * self.vt_net(nstate)
        q1 = self.q1_net(state, action)
        q2 = self.q2_net(state, action)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)

        # Pi-Loss
        pi_action, log_pi_action = self.pi_net.select_action(state)
        q = torch.min(self.q1_net(state, pi_action), self.q2_net(state, pi_action))
        pi_loss = torch.mean(log_pi_action - q / self.alpha)

        # V-Loss
        v_target = q - self.alpha * log_pi_action
        v = self.v_net(state)
        v_loss = F.mse_loss(v, v_target)

        # Vt-Loss
        vt_loss = 0.0
        for v_param, vt_param in zip(self.v_net.parameters(), self.vt_net.parameters()):
            vt_loss += 0.5 * torch.sum((vt_param - v_param)**2)
        
        return q1_loss, q2_loss, pi_loss, v_loss, vt_loss

    def get_loss_grad(self, loss, net, create_graph=False, retain_graph=None):
        grads = torch.autograd.grad(loss, net.parameters(), 
                                    create_graph=create_graph, retain_graph=retain_graph)
        loss_grad = torch.cat([grad.contiguous().view(-1) for grad in grads])
        return loss_grad
    
    def get_loss_grad_list(self, loss_list, net_list, create_graph=False):
        loss_grad_list = []
        for loss, net in zip(loss_list, net_list):
            grad = self.get_loss_grad(loss, net, create_graph)
            loss_grad_list.append(grad)
        return loss_grad_list

    def get_ht_grad_list(self, loss_list, net_list, xi_list, scale=1e-4):
        loss_grad_list = self.get_loss_grad_list(loss_list, net_list, create_graph=True)

        loss_for_ht_grad_list = 0.0
        for loss_grad, xi in zip(loss_grad_list, xi_list):
            loss_for_ht_grad_list += loss_grad @ xi * scale # loss are too big

        ht_grad_list = []
        for i, net in enumerate(net_list):
            retain_graph = True if i < len(loss_list)-1 else False
            grad = self.get_loss_grad(loss_for_ht_grad_list, net, retain_graph=retain_graph)
            ht_grad_list.append(grad)

        return tuple(ht_grad_list)
        
    def get_h_grads(self, state, action, reward, nstate, mask,
                q1_loss_grad_old, q2_loss_grad_old, pi_loss_grad_old, 
                v_loss_grad_old, vt_loss_grad_old):
        pass


