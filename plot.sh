#!/bin/bash
for env_name in HalfCheetah-v2 ; # Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2  ;
do
    python plot.py --alg_list local_trpo local_trpo2 local_trpo3 \
                              hmtrpo global_trpo local_ppo global_ppo \
                --env_name $env_name \
                --workers 2 \
                --reward_step 1
done