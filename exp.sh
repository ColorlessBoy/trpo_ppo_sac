#!/bin/bash

for seed in 0 ; # 100 200 ; 
do
    for env_name in HalfCheetah-v2 # Walker2d-v2 Ant-v2 Humanoid-v2 Swimmer-v2 HumanoidStandup-v2 Hopper-v2 ;
    do
        for reward_step in 2 3 ; # 0 1 2 3 ;
        do
        #   for alg in local_trpo local_trpo2 local_trpo3 hmtrpo global_trpo ;
        #   do
        #       cd /home/peng/Documents/trpo_ppo_sac/trpo
        #       echo "Experiment:" $alg "_" $env_name "_" $seed
        #       python parallel_main.py --env_name $env_name \
        #                               --alg $alg \
        #                               --workers 20 \
        #                               --device cuda \
        #                               --seed $seed \
        #                               --reward_step $reward_step
        #   done
            for alg in local_ppo global_ppo ;
            do
                cd /home/peng/Documents/trpo_ppo_sac/ppo
                echo "Experiment:" $alg "_" $env_name "_" $seed
                python parallel_main.py --env_name $env_name \
                                        --alg $alg \
                                        --workers 20 \
                                        --device cuda \
                                        --seed $seed \
                                        --reward_step $reward_step
            done
        done
    done
done