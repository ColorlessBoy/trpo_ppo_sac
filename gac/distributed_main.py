import gym
import torch
import numpy as np
from torch.multiprocessing import Process
import torch.distributed as dist
from time import time
import os
import csv
import json
from collections import namedtuple

from utils import ReplayBuffer, MLPActorCritic
from gac import GAC

weights=np.array([[0.6, 0.2, 0,   0,    0,      0,    0.2,  0],
                  [0.2, 0.2, 0.2, 0,    0,      0,    0.2,  0.2],
                  [0,   0.2, 0.8, 0,    0,      0,    0,    0],
                  [0,   0,   0,   0.55, 0,      0.25, 0.20, 0],
                  [0,   0,   0,   0,    0.4167, 0.25, 0,    0.3333],
                  [0,   0,   0,   0.25, 0.25,   0.3,  0.2,  0],
                  [0.2, 0.2, 0,   0.2,  0,      0.2,  0.2,  0],
                  [0,   0.2, 0,   0,    0.3333, 0,    0,    0.4667]])

def average_parameters(rank, size, params):
    for param in params:
        tensor_list = [torch.empty_like(param.data) for _ in range(size)]
        dist.all_gather(tensor_list, param.data)
        param.data.mul_(weights[rank][rank])
        for n, tensor in enumerate(tensor_list):
            if n != rank and weights[rank][n] > 0:
                param.data.add_(weights[rank][n] * tensor)

def synchronous_parameters(model):
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def run(rank, size, args):
    if "Hopper" in args.env_name:
        env = gym.make(args.env_name, terminate_when_unhealthy=False)
        test_env = gym.make(args.env_name, terminate_when_unhealthy=False)
    else:
        env = gym.make(args.env_name)
        test_env = gym.make(args.env_name)
    device = torch.device(args.device)

    # 1.Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    test_env.seed(args.seed + 999)

    # 2.Create actor, critic, EnvSampler() and PPO.
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    actor_critic = MLPActorCritic(obs_dim, act_dim, hidden_sizes=args.hidden_sizes).to(device)
    synchronous_parameters(actor_critic.actor)
    synchronous_parameters(actor_critic.critic1)
    synchronous_parameters(actor_critic.critic2)

    replay_buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size)
    obs_mean_params = [torch.from_numpy(replay_buffer.obs_mean), torch.from_numpy(replay_buffer.obs_square_mean)]
    gac = GAC(actor_critic, replay_buffer, device=device,
              alpha_start=args.alpha_start, alpha_min=args.alpha_min, alpha_max=args.alpha_max)

    # 3.Start training.
    def get_action(o, deterministic=False):
        o = torch.FloatTensor(o.reshape(1, -1)).to(device)
        a = actor_critic.act(o, deterministic)
        return a

    def test_agent():
        test_ret, test_len = 0, 0
        for j in range(args.epoch_per_test):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == args.max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _ = test_env.step(get_action(o, True) * act_limit)
                ep_ret += r
                ep_len += 1
            test_ret += ep_ret
            test_len += ep_len
        return test_ret / args.epoch_per_test, test_len / args.epoch_per_test

    total_step = args.total_epoch * args.steps_per_epoch
    o, d, ep_ret, ep_len = env.reset(), False, 0, 0
    for t in range(1, total_step+1):
        if t <= args.start_steps:
            a = env.action_space.sample() / act_limit
        else:
            a = get_action(o, deterministic=False)
        
        o2, r, d, _ = env.step(a * act_limit)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)

        d = False if ep_len==args.max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r * args.reward_scale, o2, d)
        gac.update_obs_param()

        o = o2
        if d or (ep_len == args.max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= args.update_after and t % args.steps_per_update==0:
            for j in range(args.steps_per_update):
                loss_a, loss_c, alpha = gac.update(args.batch_size)
            gac.update_beta()
            print("Rank{} Step {:>10}: loss_actor = {:<22}, loss_critic = {:<22}, alpha = {:<20}, beta = {:<20}".format(rank, t, loss_a, loss_c, alpha, gac.beta))
            average_parameters(rank, size, gac.actor_critic.actor.parameters())
            average_parameters(rank, size, gac.actor_critic.critic1.parameters())
            average_parameters(rank, size, gac.actor_critic.critic2.parameters())
            
            obs_mean_params = [torch.from_numpy(replay_buffer.obs_mean).to(device), torch.from_numpy(replay_buffer.obs_square_mean).to(device)]
            average_parameters(rank, size, obs_mean_params)
            replay_buffer.obs_mean = obs_mean_params[0].cpu().numpy()
            replay_buffer.obs_square_mean = obs_mean_params[1].cpu().numpy()

        # End of epoch handling
        if t >= args.update_after and t % args.steps_per_epoch == 0:
            test_ret, test_len = test_agent()
            print("Rank{} Step {:>10}: test_ret = {:<20}, test_len = {:<20}".format(rank, t, test_ret, test_len))
            yield t, test_ret, test_len

Args = namedtuple('Args',
               ('alg_name',
                'env_name', 
                'device', 
                'seed', 
                'hidden_sizes', 
                'buffer_size',
                'epoch_per_test',
                'max_ep_len', 
                'total_epoch', 
                'steps_per_epoch',
                'start_steps',
                'reward_scale',
                'update_after',
                'steps_per_update',
                'batch_size',
                'alpha_start',
                'alpha_min',
                'alpha_max'))

def parallel_run(rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

    alg_args = Args(args.alg,       # alg_name
                args.env,           # env_name
                args.device if (args.device == 'cpu') else 'cuda:{}'.format(rank % torch.cuda.device_count()), # device
                args.seed + rank,   # seed
                [400, 300],         # hidden_sizes
                int(1e6),           # replay buffer size
                10,                 # epoch per test
                1000,               # max_ep_len
                750,                # total epochs
                4000,               # steps per epoch
                10000,              # start steps
                args.reward_scale,  # reward scale 
                1000,               # update after
                50,                 # steps_per_update
                100,                # batch size
                args.alpha_start,
                args.alpha_min,
                args.alpha_max)

    logdir = "./data/gac-distributed/{}/{}-rank{}-seed{}-{}".format(alg_args.env_name, alg_args.env_name, rank, alg_args.seed, time())
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    config_name = 'config.json'
    file_name = 'progress.csv'

    config_json = json.dumps(alg_args._asdict())
    config_json = json.loads(config_json)
    output = json.dumps(config_json, separators=(',',':\t'), indent=4, sort_keys=True)
    with open(os.path.join(logdir, config_name), 'w') as out:
        out.write(output)

    full_name = os.path.join(logdir, file_name)
    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile, delimiter='\t')
    writer.writerow(['TotalEnvInteracts', 'AverageTestEpRet', 'AverageTestEpLen'])

    for t, reward, len in fn(rank, size, alg_args):
        writer.writerow([t, reward, len])
        csvfile.flush()

    csvfile.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--env', default='HalfCheetah-v3', metavar='G',
                        help='name of environment name (default: HalfCheetah-v3)')
    parser.add_argument('--device', default='cuda', metavar='G',
                        help='device (default cuda)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='total epochs(default: 1000)')
    parser.add_argument('--reward_scale', type=float, default=1.0, metavar='N',
                        help='reward_scale (default: 1.0)')
    parser.add_argument('--alpha_start', type=float, default=1.2, metavar='N',
                        help='alpha_start (default: 1.2)')
    parser.add_argument('--alpha_min', type=float, default=1.0, metavar='N',
                        help='alpha_min (default: 1.0)')
    parser.add_argument('--alpha_max', type=float, default=1.5, metavar='N',
                        help='alpha_max (default: 1.5)')
    parser.add_argument('--agent', type=int, default=8, metavar='N',
                        help='number of agents (default: 8)')
    parser.add_argument('--alg', default='gac-distributed', metavar='G',
                        help='algorithm (default gac-parallel')

    
    args = parser.parse_args()

    logdir = "./data/gac-distributed/{}".format(args.env)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    size = args.agent
    processes = []
    backend = 'gloo' if args.device == 'cpu' else 'nccl'

    for rank in range(size):
        p = Process(target=parallel_run, args=(rank, size, run, args, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join() # wait all process stop.