import gym
import torch
from collections import namedtuple
import os
import csv
from time import time
import random


from models import PolicyNetwork, ValueNetwork, QNetwork
from utils import EnvSampler, hard_update
from sac import SAC
from sacnp import SACNP
from sacnp2 import SACNP2

def run(args):
    env = gym.make(args.env_name)

    device = torch.device(args.device)

    # 1. Set some necessary seed.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    # 2. Create nets. 
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    v_net = ValueNetwork(state_size, args.hidden_sizes, 
        activation=args.activation).to(device)
    q1_net = QNetwork(state_size, action_size, args.hidden_sizes,
        activation=args.activation).to(device)
    q2_net = QNetwork(state_size, action_size, args.hidden_sizes,
        activation=args.activation).to(device)
    pi_net = PolicyNetwork(state_size, action_size, args.hidden_sizes,
        activation=args.activation, output_activation=args.output_activation).to(device)
    vt_net = ValueNetwork(state_size, args.hidden_sizes,
        activation=args.activation).to(device)

    # hard_update(vt_net, v_net)
    pi_path = '/home/peng/Documents/python/RL/trpo_ppo_sac/sac/models/alg_sac/env_HalfCheetah-v2/pi_net_alg_sac_env_HalfCheetah-v2_batch100_seed0_step300000_time1579368167.1438684.pth.tar'
    q1_path = '/home/peng/Documents/python/RL/trpo_ppo_sac/sac/models/alg_sac/env_HalfCheetah-v2/q1_net_alg_sac_env_HalfCheetah-v2_batch100_seed0_step300000_time1579368167.1428776.pth.tar'
    q2_path = '/home/peng/Documents/python/RL/trpo_ppo_sac/sac/models/alg_sac/env_HalfCheetah-v2/q2_net_alg_sac_env_HalfCheetah-v2_batch100_seed0_step300000_time1579368167.1433735.pth.tar'
    v_path  = '/home/peng/Documents/python/RL/trpo_ppo_sac/sac/models/alg_sac/env_HalfCheetah-v2/v_net_alg_sac_env_HalfCheetah-v2_batch100_seed0_step300000_time1579368167.1420517.pth.tar'
    vt_path = '/home/peng/Documents/python/RL/trpo_ppo_sac/sac/models/alg_sac/env_HalfCheetah-v2/vt_net_alg_sac_env_HalfCheetah-v2_batch100_seed0_step300000_time1579368167.1443515.pth.tar'
    pi_net.load_state_dict(torch.load(pi_path))
    q1_net.load_state_dict(torch.load(q1_path))
    q2_net.load_state_dict(torch.load(q2_path))
    v_net.load_state_dict(torch.load(v_path))
    vt_net.load_state_dict(torch.load(vt_path))

    env_sampler = EnvSampler(env, args.max_episode_length)

    if args.alg_name == 'sac':
        alg = SAC(v_net, q1_net, q2_net, pi_net, vt_net,
                    gamma=0.99, alpha=0.2,
                    v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = args.vt_lr,
                    device=device)
    elif args.alg_name == 'sacnp':
        alg = SACNP(v_net, q1_net, q2_net, pi_net, vt_net,
                    gamma=0.99, alpha=0.2, lm=1,
                    v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = args.vt_lr,
                    device=device)
    elif args.alg_name == 'sacnp2':
        alg = SACNP2(v_net, q1_net, q2_net, pi_net, vt_net,
                    gamma=0.99, alpha=0.2, lm=1,
                    v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = args.vt_lr,
                    device=device)
    # 3. Warmup.
    start_time = time()
    env_sampler.addSamples(args.start_steps)
    print("Warmup uses {}s.".format(time() - start_time))

    # 4. Start training.
    def get_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, _ = pi_net.select_action(state)
        return action.cpu().numpy()[0]

    def get_mean_action(state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = pi_net.get_mean_action(state)
        return action.cpu().numpy()[0]

#   sac = SAC(v_net, q1_net, q2_net, pi_net, vt_net,
#               gamma=0.99, alpha=0.2,
#               v_lr=1e-3, q_lr=1e-3, pi_lr=1e-3, vt_lr = args.vt_lr,
#               device=device)

#   trajectory = 0
    warm_steps = 500000
#   for step in range(1, warm_steps+1):
#       done, episode_reward = env_sampler.addSample(get_action)
#       batch = env_sampler.sample(args.batch_size)
#       losses = sac.update(*batch)

#       if done:
#           trajectory += 1

#       test_reward = None
#       test_nets = None
#       if done or step == args.total_steps:
#           if trajectory % args.test_frequency == 0 or step == args.total_steps:
#               test_reward = env_sampler.test(get_mean_action, 10)
#           yield (step, episode_reward, *losses, test_reward, test_nets)

    trajectory = 0
    for step in range(warm_steps+1, warm_steps+args.total_steps+1):
        done, episode_reward = env_sampler.addSample(get_action)
        if args.alg_name == 'sac':
            batch = env_sampler.sample(args.batch_size)
            losses = alg.update(*batch)
        else:
            batch1 = env_sampler.sample(args.batch_size)
            batch2 = env_sampler.sample(args.batch_size)
            losses = alg.update(batch1, batch2)
        if done:
            trajectory += 1

        test_reward = None
        test_nets = None
        if done or step == args.total_steps:
            if trajectory % args.test_frequency == 0 or step == args.total_steps:
                test_reward = env_sampler.test(get_mean_action, 10)
            if step == args.total_steps:
                test_nets = (v_net, q1_net, q2_net, pi_net, vt_net)
            yield (step, episode_reward, *losses, test_reward, test_nets)

# The properties of args:
# 0. alg_name (default: sac)
# 1. env_name (default: HalfCheetah-v2)
# 2. device (default: cpu)
# 3. hidden_sizes (default: (64, 64))
# 4. batch_size (default：256)
# 5. total_steps (default: 1000000)
# 6. max_episode_length (default: 1000)
# 7. start_steps (default: 10000)
# 8. seed (default: 0)
# 9. test_frequency (default: 10)
# 10. model_save_frequency (default: 100)
# 11. activation (default: torch.relu)
# 12. output_activation (default: torch.tanh)

Args = namedtuple( 'Args',
    ('alg_name',
    'env_name',
    'device',
    'hidden_sizes',
    'batch_size',
    'total_steps',
    'max_episode_length',
    'start_steps',
    'seed',
    'vt_lr',
    'test_frequency',
    'model_save_frequency',
    'activation',
    'output_activation')
)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run experiment with optional args')
    parser.add_argument('--alg', default='sac', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--env_name', default='HalfCheetah-v2', metavar='G',
                        help='name of environment name (default: HalfCheetah-v2)')
    parser.add_argument('--device', default='cpu', metavar='G',
                        help='device (default cpu)')
    parser.add_argument('--total_steps', type=int, default=100000, metavar='N',
                        help='total_steps (default 100000)')
    parser.add_argument('--test_frequency', type=int, default=10, metavar='N',
                        help='test_frequency (default 10)')
    parser.add_argument('--model_save_frequency', type=int, default=100, metavar='N',
                        help='model save frequency (default 10)')
    args = parser.parse_args()

    alg_args = Args(
        args.alg,         # alg_name
        args.env_name,    # env_name
        args.device,      # device
        (400, 300),       # hidden_size
        100,              # batch_size
        args.total_steps, # total_steps
        1000,             # max_episode_length
        10000,            # start_steps
        args.seed,        # seed
        0.005,            # vt_lr
        args.test_frequency,       # test_frequency
        args.model_save_frequency, # model_save_frequency
        torch.relu,       # activation
        torch.tanh,       # output_activation
    )

    # Train Dir
    train_logdir = "./logs/alg_{}/env_{}".format(alg_args.alg_name, alg_args.env_name)
    file_name = 'train_alg_{}_env_{}_batch{}_seed{}_total_steps{}_time{}.csv'.format(alg_args.alg_name, 
                alg_args.env_name, alg_args.batch_size, alg_args.seed, alg_args.total_steps, time())
    if not os.path.exists(train_logdir):
        os.makedirs(train_logdir)
    full_name = os.path.join(train_logdir, file_name)
    csvfile = open(full_name, 'w')
    writer = csv.writer(csvfile)
    writer.writerow(['step', 'reward'])

    # Test Dir
    test_logdir = "./logs/alg_{}/env_{}".format(alg_args.alg_name, alg_args.env_name)
    test_file_name = 'test_alg_{}_env_{}_batch{}_seed{}_total_steps{}_time{}.csv'.format(alg_args.alg_name, 
                    alg_args.env_name, alg_args.batch_size, alg_args.seed, alg_args.total_steps, time())
    if not os.path.exists(test_logdir):
        os.makedirs(test_logdir)
    test_full_name = os.path.join(test_logdir, test_file_name)
    test_csvfile = open(test_full_name, 'w')
    test_writer = csv.writer(test_csvfile)
    test_writer.writerow(['step', 'reward'])

    # Model Dir
    modeldir = "./models/alg_{}/env_{}".format(alg_args.alg_name, alg_args.env_name)
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
    
    start_time = time()

    for step, reward, q1_loss, q2_loss, pi_loss, v_loss, vt_loss, test_reward, test_nets in run(alg_args):
        writer.writerow([step, reward])
        print("Step {}: Reward = {:>10.6f}, q1_loss = {:>8.6f}, q2_loss = {:>8.6f}, pi_loss = {:>8.6f}, v_loss = {:>8.6f}, vt_loss = {:>8.6f}".format(
            step, reward, q1_loss, q2_loss, pi_loss, v_loss, vt_loss
        ))

        if test_reward is not None:
            test_writer.writerow([step, test_reward])
            print("Step {}: Test reward = {}".format(step, test_reward))
        if test_nets is not None:
            nets_name = ('v_net', 'q1_net', 'q2_net', 'pi_net', 'vt_net')
            for name, net in zip(nets_name, test_nets):
                modelfile = "{}_alg_{}_env_{}_batch{}_seed{}_step{}_time{}.pth.tar".format(name, alg_args.alg_name, 
                        alg_args.env_name, alg_args.batch_size, alg_args.seed, step, time())
                model_full_name = os.path.join(modeldir, modelfile)
                torch.save(net.state_dict(), model_full_name)

    print("Total time: {}s.".format(time() - start_time))