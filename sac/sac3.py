import torch
from torch.optim import SGD, SGD
import torch.nn.functional as F

class SAC3(object):
    def __init__(self, v_net, q1_net, q2_net, pi_net, vt_net, 
                gamma=0.99, alpha=0.2,
                v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr=1e-3,
                device=torch.device('cpu')):
        # nets
        self.v_net, self.q1_net, self.q2_net, self.pi_net, self.vt_net = \
            v_net, q1_net, q2_net, pi_net, vt_net

        # hyperparameters
        self.gamma, self.alpha = gamma, alpha

        # device
        self.device = device

        # optimization
        self.v_optim  = SGD(self.v_net.parameters(),  lr = v_lr )
        self.q1_optim = SGD(self.q1_net.parameters(), lr = q_lr )
        self.q2_optim = SGD(self.q2_net.parameters(), lr = q_lr)
        self.pi_optim = SGD(self.pi_net.parameters(), lr = pi_lr)
        self.vt_optim = SGD(self.vt_net.parameters(), lr = vt_lr)

    def update(self, batch1):
        def to_device(batch):
            state  = torch.FloatTensor(batch[0]).to(self.device)
            action = torch.FloatTensor(batch[1]).to(self.device)
            reward = torch.FloatTensor(batch[2]).to(self.device).unsqueeze(1)
            nstate = torch.FloatTensor(batch[3]).to(self.device)
            mask   = torch.FloatTensor(batch[4]).to(self.device).unsqueeze(1)
            return state, action, reward, nstate, mask
        
        batch1 = to_device(batch1) 

        net_list = (self.q1_net, self.q2_net, self.pi_net, self.v_net, self.vt_net)
        net_optim_list = (self.q1_optim, self.q2_optim, self.pi_optim, self.v_optim, self.vt_optim)

        loss_list = self.get_loss_list(*batch1)

        for loss in loss_list:
            loss_grad_list = self.get_loss_grad_list(loss, net_list)
            for net, net_optim, loss_grad in zip(net_list, net_optim_list, loss_grad_list):
                if loss_grad is not None:
                    prev_ind = 0
                    for param in net.parameters():
                        flat_size = param.numel()
                        param.grad = \
                            loss_grad[prev_ind:prev_ind + flat_size].view(param.size())
                        prev_ind += flat_size
                    net_optim.step()

        return loss_list

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
        pi_action_v, log_pi_action_v = self.pi_net.select_action(state)
        q_v = torch.min(self.q1_net(state, pi_action_v), self.q2_net(state, pi_action_v))
        v_target = q_v - self.alpha * log_pi_action_v
        v = self.v_net(state)
        v_loss = F.mse_loss(v, v_target)

        # Vt-Loss
        vt_loss = 0.0
        for v_param, vt_param in zip(self.v_net.parameters(), self.vt_net.parameters()):
            vt_loss += 0.5 * torch.sum((vt_param - v_param)**2)
        
        return q1_loss, q2_loss, pi_loss, v_loss, vt_loss

    def get_loss_grad(self, loss, net, retain_graph=None):
        grads = torch.autograd.grad(loss, net.parameters(), retain_graph=retain_graph, allow_unused=True)
        loss_grad = None
        if grads[0] is not None:
            loss_grad = torch.cat([grad.contiguous().view(-1) for grad in grads])
            loss_grad = loss_grad / (loss_grad.norm() + 1e-6)
        return loss_grad
    
    def get_loss_grad_list(self, loss, net_list):
        loss_grad_list = []
        for net in net_list:
            grad = self.get_loss_grad(loss, net, True)
            loss_grad_list.append(grad)
        return loss_grad_list