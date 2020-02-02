#!/bin/bash

for seed in 0 ;
do
    for env_name in Humanoid-v2 ;
    do
        for reward_step in 2 ;
        do
            for alg in global_ppo ;
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