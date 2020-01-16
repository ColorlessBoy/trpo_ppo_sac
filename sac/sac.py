import torch
from torch.optim import Adam, SGD
import torch.nn.functional as F

from utils import soft_update

class SAC(object):
    def __init__(self, v_net, q1_net, q2_net, pi_net, vt_net, 
                gamma=0.99, alpha=0.2,
                v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = 5e-3,
                device=torch.device('cpu')):
        # nets
        self.v_net, self.q1_net, self.q2_net, self.pi_net, self.vt_net = \
            v_net, q1_net, q2_net, pi_net, vt_net

        # hyperparameters
        self.gamma, self.alpha = gamma, alpha

        # device
        self.device = device

        # optimization
        self.v_optim  = Adam(self.v_net.parameters(),  lr = v_lr )
        self.q1_optim = Adam(self.q1_net.parameters(), lr = q_lr )
        self.q2_optim = Adam(self.q2_net.parameters(), lr = q_lr)
        self.pi_optim = Adam(self.pi_net.parameters(), lr = pi_lr)
        self.vt_optim = SGD(self.vt_net.parameters(),  lr = vt_lr)

        self.vt_lr = vt_lr

    def update(self, state, action, reward, nstate, mask):
        state  = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        nstate = torch.FloatTensor(nstate).to(self.device)
        mask   = torch.FloatTensor(mask).to(self.device).unsqueeze(1)

        q1_loss, q2_loss = self.update_q1_net(state, action, reward, nstate, mask)
        pi_loss = self.update_pi_net(state)
        v_loss  = self.update_v_net(state)
        vt_loss = self.update_vt_net(state)

        return q1_loss, q2_loss, pi_loss, v_loss, vt_loss
    
    def update_q1_net(self, state, action, reward, nstate, mask):
        with torch.no_grad():
            q1_target = reward + self.gamma * mask * self.vt_net(nstate)
        q = self.q1_net(state, action)
        q2 = self.q2_net(state, action)
        q1_loss = F.mse_loss(q, q1_target)
        q2_loss = F.mse_loss(q2, q1_target)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()
        return q1_loss, q2_loss
    
    def update_pi_net(self, state):
        action, log_pi_action = self.pi_net.select_action(state)

        q = torch.min(self.q1_net(state, action), self.q2_net(state, action))

        pi_loss = torch.mean(log_pi_action - q / self.alpha)

        self.pi_optim.zero_grad()
        pi_loss.backward()
        self.pi_optim.step()

        return pi_loss
    
    def update_v_net(self, state):
        with torch.no_grad():
            action, log_pi_action = self.pi_net.select_action(state)
            q = torch.min(self.q1_net(state, action), self.q2_net(state, action))
            v_target = q - self.alpha * log_pi_action

        v = self.v_net(state)
        v_loss = F.mse_loss(v, v_target)

        self.v_optim.zero_grad()
        v_loss.backward()
        self.v_optim.step()

        return v_loss
    
    def update_vt_net(self, state):
        vt_loss = 0.0
        for v_param, vt_param in zip(self.v_net.parameters(), self.vt_net.parameters()):
            vt_loss += 0.5 * torch.sum((v_param.detach() - vt_param)**2)

        self.vt_optim.zero_grad()
        vt_loss.backward()
        self.vt_optim.step()

        return vt_loss