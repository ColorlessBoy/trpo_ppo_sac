#!/bin/bash
rm /home/peng/anaconda3/envs/pytorch/lib/python3.6/site-packages/mujoco_py/generated/*.lock
cd /home/peng/Documents/trpo_ppo_sac/trpo
python parallel_main.py --env_name HalfCheetah-v2 \
                    --alg global_trpo \
                    --workers 16 \
                    --device cuda \
                    --seed 100 \
                    --reward_step 2 