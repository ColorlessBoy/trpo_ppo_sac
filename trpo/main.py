import gym
import torch
import os
import csv
from time import time
from collections import namedtuple
import json
import numpy as np

from utils import EnvSampler
from models import PolicyNetwork, ValueNetwork
from trpo import TRPO


# The properties of args:
# 1. env_name (default = 'HalfCheetah-v2')
# 2. device (default = 'cuda:0')
# 3. seed (default = 1)
# 4. hidden_sizes (default = (64, 64))
# 5. max_episode_step (default = 1000)
# 6. batch_size (default = 1000)
# 7. episodes (default = 1000)
# 8. value_lr (default = 1e-3)
# 9. value_steps_per_update (default=80)
# 10. cg_steps (default = 20)
# 11. lineasearch_steps (default = 20)
# 12. gamma (default = 0.99)
# 13. tau (default = 0.97)
# 14. damping (default = 0.1)
# 15. max_kl (default = 0.01)

def main(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    device = torch.device(args.device)

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    test_env.seed(args.seed + 999)

    # 2.Create actor, critic, EnvSampler() and TRPO.
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, 
                hidden_sizes=args.hidden_sizes, init_std=args.init_std).to(device)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes).to(device)
    env_sampler = EnvSampler(env, args.max_episode_step, args.reward_scale)
    trpo = TRPO(actor, 
                critic,
                args.value_lr,
                args.value_steps_per_update,
                args.cg_steps,
                args.linesearch_steps,
                args.gamma,
                args.tau,
                args.damping,
                args.max_kl,
                device)
        
    def get_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor.select_action(state)
        return action.cpu().numpy()[0]

    def test_agent():
        test_ret, test_len = 0, 0
        for j in range(10):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == args.max_episode_step)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(env_sampler.action_decode(get_action(o)))
                ep_ret += r
                ep_len += 1
            test_ret += ep_ret
            test_len += ep_len
        return test_ret / 10, test_len / 10

    def get_value(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            value = critic(state)
        return value.cpu().numpy()[0, 0]

    total_step = 0
    for _ in range(1, args.episodes+1):
        _, samples = env_sampler(get_action, args.batch_size, get_value)
        actor_loss, value_loss = trpo.update(*samples)
        total_step += args.batch_size

        print("loss_actor = {:<22}, loss_critic = {:<22}".format(actor_loss, value_loss))

        test_ret, test_len = test_agent()
        yield total_step, test_ret, test_len

Args = namedtuple('Args', 
                    ('alg_name',
                    'env_name',
                    'device',
                    'seed',
                    'hidden_sizes',
                    'max_episode_step',
                    'batch_size',
                    'episodes',
                    'value_lr',
                    'value_steps_per_update',
                    'cg_steps',
                    'linesearch_steps',
                    'gamma',
                    'tau',
                    'damping',
                    'max_kl',
                    'init_std',
                    'reward_scale'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--batch', type=int, default=4000, metavar='N',
                        help='number of batch size (default: 4000)')
    parser.add_argument('--episodes', type=int, default=7500, metavar='N',
                        help='number of eperiment episodes(default: 7500)')
    parser.add_argument('--env_name', default='HalfCheetah-v2', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cuda', metavar='G',
                        help='device (default cuda)')
    parser.add_argument('--reward_scale', type=float, default=1.0, metavar='N',
                        help='reward scale(default 1.0)')
    
    args = parser.parse_args()
    alg_args = Args("trpo",         # alg_name
                args.env_name,      # env_name
                args.device,        # device
                args.seed,          # seed
                (400, 300),         # hidden_sizes
                1000,               # max_episode_step
                args.batch,         # batch_size
                args.episodes,      # episodes
                1e-3,               # value_lr
                50,                 # value_steps_per_update
                20,                 # cg_steps
                20,                 # linesearch_steps
                0.995,              # gamma
                0.97,               # tau
                0.1,                # damping
                0.02,               # max_kl
                1.0,                # init_std
                args.reward_scale)  # reward scale
                
    logdir = "./data/trpo/{}/{}-seed{}-{}".format(alg_args.env_name, alg_args.env_name,alg_args.seed, time())
    config_name = 'config.json'
    file_name = 'progress.csv'
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config_json = json.dumps(alg_args._asdict())
    config_json = json.loads(config_json)
    output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
    with open(os.path.join(logdir, config_name), 'w') as out:
        out.write(output)

    full_name = os.path.join(logdir, file_name)

    full_name = os.path.join(logdir, file_name)
    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['TotalEnvInteracts', 'AverageTestEpRet', 'AverageTestEpLen'])

    for step, reward, length in main(alg_args):
        writer.writerow([step, reward, length])
        csvfile.flush()
