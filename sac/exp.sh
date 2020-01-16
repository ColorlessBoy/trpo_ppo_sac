#!/bin/bash
python main.py --device cuda --total_steps 1000000
python main_np.py --device cuda --total_steps 1000000