#!/bin/bash

for seed in 0 100 200 ; 
do
    for env_name in HalfCheetah-v2 Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2 ;
    do
        for alg in local_trpo local_trpo2 local_trpo3 hmtrpo global_trpo ;
        do
            cd /home/peng/Documents/rl_algorithms/trpo
            echo $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --agent 12 \
                                    --device cuda \
                                    --seed $seed
        done
        for alg in local_ppo global_ppo ;
        do
            cd /home/peng/Documents/rl_algorithms/ppo
            echo $alg "_" $env_name "_" $seed
            python parallel_main.py --env_name $env_name \
                                    --alg $alg \
                                    --agent 12 \
                                    --device cuda \
                                    --seed $seed
        done
    done
done