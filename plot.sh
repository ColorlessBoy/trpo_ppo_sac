#!/bin/bash
for env_name in Navigation2DEnv-FL ;
do
    for alg in local_trpo3 hmtrpo global_trpo local_ppo global_ppo ;
    do
            python plot2.py --alg_list $alg \
                        --env_name $env_name \
                        --workers 4 \
                        --keywords reward_step_0 reward_step_1 reward_step_2
    done
done