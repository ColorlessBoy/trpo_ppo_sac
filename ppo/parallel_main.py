import os
import gym
import torch
from torch.multiprocessing import Process
import torch.distributed as dist
from time import time
import csv
from collections import namedtuple
import json
import numpy as np

from utils import EnvSampler
from models import PolicyNetwork, ValueNetwork
from local_ppo import LocalPPO
from global_ppo import GlobalPPO
from distributed_ppo import DistributedPPO

# The properties of args:
# 1. env_name (default = 'HalfCheetah-v2')
# 2. device (default = "cuda:0")
# 3. seed (default = 1)
# 4. hidden_sizes (default = (64, 32))
# 5. episodes (default = 100. Not the number of trajectories, but the number of batches.)
# 6. max_episode_step (default = 1000)
# 7. batch_size (default = 4000)
# 8. gamma (default = 0.99)
# 9. tau (default = 0.97)
# 10. clip (default = 0.2)
# 11. max_kl (default =  0.01)
# 12. pi_steps_per_update (default = 80) 
# 13. value_steps_per_update (default = 80)
# 14. pi_lr (default = 3e-4)
# 15. value_lr (default = 1e-3)
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
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    actor = PolicyNetwork(state_size, action_size, hidden_sizes=args.hidden_sizes).to(device)
    critic = ValueNetwork(state_size, hidden_sizes=args.hidden_sizes).to(device)
    env_sampler = EnvSampler(env, args.max_episode_step, args.reward_scale)
    ppo_args = {
        'actor': actor,
        'critic': critic,
        'clip': args.clip,
        'gamma': args.gamma,
        'tau': args.tau,
        'target_kl': args.target_kl,
        'device': device,
        'pi_steps_per_update': args.pi_steps_per_update,
        'value_steps_per_update': args.value_steps_per_update,
        'pi_lr': args.pi_lr,
        'v_lr': args.value_lr
    }
    if args.alg_name == 'ppo-parallel':
        alg = LocalPPO(**ppo_args)
    elif args.alg_name == 'ppo-distributed':
        alg = DistributedPPO(**ppo_args)
    else:
        alg = GlobalPPO(**ppo_args)

    def get_action(state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor.select_action(state)
        return action.detach().cpu().numpy()[0]

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

    # 3.Start training.
    total_step = 0
    for _ in range(1, args.episodes+1):
        _, samples = env_sampler(get_action, args.batch_size, get_value)
        actor_loss, value_loss = alg.update(*samples)
        total_step += args.batch_size

        print("Rank{}: loss_actor = {:<22}, loss_critic = {:<22}".format(rank, actor_loss, value_loss))

        test_ret, test_len = test_agent()
        yield total_step, test_ret, test_len

Args = namedtuple('Args',
                ('alg_name',
                'env_name', 
                'device', 
                'seed', 
                'hidden_sizes', 
                'episodes', 
                'max_episode_step', 
                'batch_size', 
                'gamma', 
                'tau', 
                'clip', 
                'target_kl',
                'pi_steps_per_update',
                'value_steps_per_update',
                'pi_lr',
                'value_lr',
                'reward_scale'))

def parallel_run(rank, size, fn, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)


    alg_args = Args(args.alg_name,  # alg_name
                args.env_name,      # env_name
                args.device if (args.device == 'cpu') else 'cuda:{}'.format(rank % torch.cuda.device_count()), # device
                args.seed + rank,   # seed
                (400, 300),         # hidden_sizes
                args.episodes,      # episodes
                1000,               # max_episode_step
                args.batch,         # batch_size
                0.995,              # gamma
                0.97,               # tau
                0.2,                # clip
                0.015,              # target_kl
                80,                 # pi_steps_per_update
                50,                 # value_steps_per_update
                3e-4,               # pi_lr
                1e-3,               # value_lr
                args.reward_scale)  # reward scale

    logdir = "./data/{}/{}/{}-rank{}-seed{}-{}".format(args.alg_name, alg_args.env_name, alg_args.env_name, rank, alg_args.seed, time())
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
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--agent', type=int, default=8, metavar='N',
                        help='number of agents (default: 8)')
    parser.add_argument('--alg_name', default="ppo-distributed", metavar='G',
                        help='name of the algorithm to run (default: ppo-distributed)')
    parser.add_argument('--env_name', default="HalfCheetah-v3", metavar='G',
                        help='name of environment to run (default: HalfCheetah-v3)')
    parser.add_argument('--batch', type=int, default=4000, metavar='N',
                        help='number of batch size (default: 4000)')
    parser.add_argument('--episodes', type=int, default=7500, metavar='N',
                        help='number of experiment episodes(default: 1000)')
    parser.add_argument('--device', default='cuda', metavar='G',
                        help='device (default: cuda)')
    parser.add_argument('--reward_scale', type=float, default=1.0, metavar='N',
                        help='reward scale (default: float)')
    args = parser.parse_args()

    logdir = "./data/{}/{}".format(args.alg_name, args.env_name)
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
