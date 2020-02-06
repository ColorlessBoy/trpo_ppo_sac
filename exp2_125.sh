#!/bin/bash

for seed in 100 ;
do
    for env_name in HalfCheetah-v2 ;
    do
        for alg in local_ppo ;
        do
            rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
            cd /home/peng/Documents/trpo_ppo_sac/ppo
            echo "Experiment:" $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --workers 64 \
                                    --device cuda \
                                    --seed $seed \
                                    --reward_step 0 1 2 3 \
                                    --master_addr 10.214.192.126 \
                                    --node_size 2 \
                                    --node_rank 1
        done
    done
done