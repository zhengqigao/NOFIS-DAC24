# NOFIS-DAC24

## Introduction 

This repo contains our code for the DAC'24 paper titled 'NOFIS: Normalizing Flow for Rare Circuit Failure Analysis'. It utilizes Normalizing Flows (NFs) and importance sampling (IS) to estimate the occurrence probability of a rare event. In this repo, we provide the code implementation for our NOFIS approach on all synthetic experiments.  The implementations for baseline methods can be requested by contacting the authors, if needed. Moreover, as the real circuit examples are proprietary and might have license issues, we have omitted them. 

To justify the reproducibility of our results, please run the code on the synthetic testbench repeatedly with 20 different seeds and check if the mean results are similar to the reported in our paper. 

## Requirements

`Python` and `Pytorch` are needed to run the code. Our code is tested on `Python 3.9.16` and `Pytorch 1.12.0`. Other versions should work well but are not tested.

## Quick Usage

To reproduce the numerical results, please check the `./exp/exp1.sh`, or directly run the following command:

```bash
# notice that the results will be directly stored under $res
$res=./results/case101/
python main_nofis.py --gpu 0 --testcase 101 --plt_fig save --save_data 1 --save_path $res
```

To reproduce the 2D visualization results, please check the `./exp/exp_visual.sh`. We have prepared a few pretrained models under the `results` folder, so that now you can directly use `./exp/exp_visual.sh` to do visualizations. In addition, please feel free to try running `./exp/exp2.sh` yourself for obtaining extra results under different random seeds obtain extra visualizations.