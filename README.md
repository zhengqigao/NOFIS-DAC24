# NOFIS-DAC24

## Introduction 

This repo contains the code for our DAC'24 paper titled 'NOFIS: Normalizing Flow for Rare Circuit Failure Analysis'. NOFIS utilizes Normalizing Flows (NFs) and importance sampling (IS) to estimate the occurrence probability of a rare event (e.g., probability smaller than 1E-4). For estimating such small probabilities, the direct Monte Carlo (MC) method is inefficient. In contrast, NOFIS can provide a more accurate and efficient estimation by learning an optimal proposal distribution with NFs, and next using it in combination with IS. 

In this repo, we provide the code implementation for our NOFIS approach on our synthetic experiments. The baseline implementations can be requested by contacting the authors, if needed. Moreover, as the circuit examples are proprietary and might have license issues, we have omitted them here.

## Requirements

`Python` and `Pytorch` are needed to run the code. Our code is tested on `Python 3.9.16` and `Pytorch 1.12.0`. Other versions should work well but are not comprehensively tested.

## Quick Usage

To reproduce the numerical results, please check the `./exp/exp1.sh`, or as an example, directly run the following command:

```bash
# notice that the results will be stored under $res
$res=./results/case101/
python main_nofis.py --gpu 0 --testcase 101 --plt_fig save --save_data 1 --save_path $res
```

To reproduce the 2D visualizations, please check the `./exp/exp_visual.sh`. We have prepared a few pretrained models under the `results` folder, so that you can directly use `./exp/exp_visual.sh` to generate visualizations. Please feel free to run `./exp/exp2.sh` yourself obtaining extra results under different random seeds and then using them to conduct visualizations, which can give you a feeling about the robustness of NOFIS. 

## Citation

Please cite our paper if you find our work useful for your research:

```
@inproceedings{zhengqi24nofis,
  title={NOFIS: Normalizing Flow for Rare Circuit Failure Analysis},
  author={Zhengqi Gao, Dinghuai Zhang, Luca Daniel, and Duane S. Boning},
  booktitle={2024 61st ACM/IEEE Design Automation Conference (DAC)},
  year={2024}
}
```
