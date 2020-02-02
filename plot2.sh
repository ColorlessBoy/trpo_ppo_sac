#!/bin/bash
for env_name in Ant-v2 ;# HalfCheetah-v2 ;# Humanoid-v2 ;# HalfCheetah-v2 ; # Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2  ;
do
    for alg in local_trpo3 hmtrpo global_trpo local_ppo global_ppo ;
    do
            python plot2.py --alg_list $alg \
                        --env_name $env_name \
                        --workers 48 16 \
                        --keywords reward_step_0 reward_step_1 reward_step_2
    done
done