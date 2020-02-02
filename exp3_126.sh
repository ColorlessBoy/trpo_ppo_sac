#!/bin/bash

for seed in 0 ;
do
    for env_name in Ant-v2 ;
    do
        for reward_step in 0 1 2 ;
        do
            for alg in local_trpo3 hmtrpo global_trpo ;
            do
                rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
                cd /home/peng/Documents/trpo_ppo_sac/trpo
                echo "Experiment:" $alg "_" $env_name "_" $seed
                python parallel_main.py --env_name $env_name \
                                        --alg $alg \
                                        --workers 16 \
                                        --device cuda \
                                        --seed $seed \
                                        --reward_step $reward_step
            done
            for alg in local_ppo global_ppo ;
            do
                rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
                cd /home/peng/Documents/trpo_ppo_sac/ppo
                echo "Experiment:" $alg "_" $env_name "_" $seed
                python parallel_main.py --env_name $env_name \
                                        --alg $alg \
                                        --workers 16 \
                                        --device cuda \
                                        --seed $seed \
                                        --reward_step $reward_step
            done
        done
    done
done