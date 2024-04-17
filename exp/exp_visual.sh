#!/bin/bash

#num_sample="1e8"

#
python get_visual2d.py --seed 20 --testcase 2 --res_path ./case2 --save --not_show --num_samples 5e5 --loss_plot --middle_plot
python get_visual2d.py --seed 23 --testcase 3 --res_path ./case3 --save --not_show  --num_samples 5e5
python get_visual2d.py --seed 5 --testcase 4 --res_path ./case4 --save --not_show --num_samples 5e5
python get_visual2d.py --seed 2 --testcase 5 --res_path ./case5 --save --not_show --num_samples 5e5

