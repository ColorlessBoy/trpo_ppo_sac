from collections import namedtuple
import random
# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', 
            ('state', 'action', 'reward', 'next_state', 'mask'))

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Memory(object):
    def __init__(self, capacity=1e6):
        self.capacity = int(capacity)
        self.position = 0
        self.memory = []

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)

class EnvSampler(object):
    def __init__(self, env, max_episode_step=1000, capacity=1e6):
        self.env = env
        self.max_episode_step = max_episode_step
        self.action_scale = (env.action_space.high - env.action_space.low)/2
        self.action_bias = (env.action_space.high + env.action_space.low)/2
        self.memory = Memory(capacity)

        self.env_init()
    
    def env_init(self):
        self.state = self.env.reset()
        self.done = False
        self.episode_step = 0
        self.episode_reward = 0.0

    # action_encode and action_decode project action into [-1, 1]^n
    def _action_encode(self, action):
        return (action - self.action_bias)/self.action_scale
    
    def _action_decode(self, action_):
        return action_ * self.action_scale + self.action_bias

    def addSample(self, get_action):
        action_ = get_action(self.state)
        action =self._action_decode(action_)
        next_state, reward, self.done, _ = self.env.step(action) 
        mask = 1.0
        if self.done or self.episode_step >= self.max_episode_step:
            mask = 0.0
        self.memory.push(self.state, action_, reward, next_state, mask)

        self.episode_reward += reward
        self.episode_step += 1
        self.state = next_state

        if mask == 0.0: 
            episode_reward = self.episode_reward
            self.env_init()
            return True, episode_reward

        return False, self.episode_reward
    
    def addSamples(self, steps):
        # Warmup the memory.
        self.env_init()
        for _ in range(steps):
            action = self.env.action_space.sample()
            action_ = self._action_encode(action)
            next_state, reward, self.done, _ = self.env.step(action) 
            mask = 1.0
            if self.done or self.episode_step >= self.max_episode_step:
                mask = 0.0
            self.memory.push(self.state, action_, reward, next_state, mask)
            self.episode_reward += reward
            self.episode_step += 1
            self.state = next_state
            if mask == 0.0: self.env_init()
        self.env_init()

    def sample(self, batch_size):
        return self.memory.sample(batch_size)
    
    def test(self, get_action, times=10):
        episode_reward = 0.0
        for _ in range(times):
            self.env_init()
            while(not self.done):
                action_ = get_action(self.state)
                action =self._action_decode(action_)
                next_state, reward, self.done, _ = self.env.step(action) 
                if self.episode_step >= self.max_episode_step:
                    self.done = True
                self.episode_reward += reward
                self.episode_step += 1
                self.state = next_state
            episode_reward += self.episode_reward
        self.env_init()
        return episode_reward / times