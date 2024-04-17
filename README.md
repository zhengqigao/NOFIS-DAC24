# NOFIS-DAC24

## Introduction 

This repo contains our code for the DAC'24 paper titled 'NOFIS: Normalizing Flow for Rare Circuit Failure Analysis'. It utilizes Normalizing Flows (NFs) and importance sampling (IS) to estimate the occurrence probability of a rare event. In this repo, we provide the code implementation for our NOFIS approach on all synthetic experiments.  The implementations for baseline methods can be requested by contacting the authors, if needed. Moreover, as the real circuit examples are proprietary and might have license issues, we have omitted them. 

To justify the reproducibility of our results, please run the code on the synthetic testbench repeatedly with 20 different seeds and check if the mean results are similar to the reported in our paper. 

## Requirements

`Python` and `Pytorch` are needed to run the code. Our code is tested on `Python 3.9.16` and `Pytorch 1.12.0`. Other versions should work well but are not tested.

## Quick Usage

```bash
python main_nofis.py --gpu 0 --testcase 101 --plt_fig save --save_data 1 --save_path ./results/
```

Please check the `exp` folder for more details.