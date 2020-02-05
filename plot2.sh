#!/bin/bash
for env_name in HalfCheetah-v2 ;
do
    for alg in local_trpo3 hmtrpo global_trpo local_ppo global_ppo;
    do
            python plot2.py --alg_list $alg \
                        --env_name $env_name \
                        --workers 64 16 \
                        --keywords reward_step_0 reward_step_1 reward_step_2 reward_step_3
    done
done