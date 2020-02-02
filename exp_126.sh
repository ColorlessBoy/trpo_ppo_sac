#!/bin/bash

for seed in 0 ; # 100 200 ; 
do
    for env_name in Humanoid-v2 ;# HalfCheetah-v2 # Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2 ;
    do
        for alg in local_trpo3 hmtrpo global_trpo ;
        do
            rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
            cd /home/peng/Documents/trpo_ppo_sac/trpo
            echo "Experiment:" $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --workers 48 \
                                    --device cuda \
                                    --seed $seed \
                                    --reward_step 0 1 2 \
                                    --master_addr 10.214.192.126 \
                                    --node_size 2 \
                                    --node_rank 0
        done
        for alg in local_ppo global_ppo ;
        do
            rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
            cd /home/peng/Documents/trpo_ppo_sac/ppo
            echo "Experiment:" $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --workers 48 \
                                    --device cuda \
                                    --seed $seed \
                                    --reward_step 0 1 2 \
                                    --master_addr 10.214.192.126 \
                                    --node_size 2 \
                                    --node_rank 0
        done
    done
done