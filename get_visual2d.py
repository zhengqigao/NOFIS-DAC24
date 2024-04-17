import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.helper import set_seed, torch2np, indicator
from utils.const import range_visualization, goldenresult_dict
from utils.testcase import get_testcase
import json
from utils.model import CTFlow
import matplotlib.pyplot as plt
import pickle
import math
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter, FormatStrFormatter

font_size = 24
line_width = 2
num_ticks = 4
cmap = 'BuGn'  # 'viridis' # 'BuGn' # 'Pastel2'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed to identify the result file')
    parser.add_argument('--gpu', type=int, default=-1, help='device id')
    parser.add_argument('--testcase', type=int, default=103, help='the index of testbench')
    parser.add_argument('--res_path', default='./results/case103', help='path to the result file')
    parser.add_argument('--margin', default=0.8, help='ratio to add to the figure')
    parser.add_argument('--box_plot', action='store_true', default=False, help='plot the box plot figure')
    parser.add_argument('--middle_plot', action='store_true', default=False, help='plot the middle distribution')
    parser.add_argument('--loss_plot', action='store_true', default=True, help='plot the loss')
    parser.add_argument('--save', action='store_true', default=False, help='save the figure')
    parser.add_argument('--not_show', action='store_true', default=False, help='do not show the figure')
    parser.add_argument('--num_samples', type=str, default='5e5')

    args = parser.parse_args()

    set_seed(0)
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))

    testcase_dict = get_testcase(args.testcase, device)

    print("### Case = %d, result path=%s, seed = %d" % (args.testcase, args.res_path, args.seed))

    for file_name in os.listdir(args.res_path):
        if file_name.endswith('json') and file_name.startswith('config'):
            config = os.path.join(args.res_path, file_name)
            break
    else:
        print("The config file is not found, will use those under ./configs")
        config = './configs/config' + str(args.testcase) + '.json'

    with open(config) as json_file:
        configs = json.load(json_file)
        configs['threshold'] = torch.Tensor(configs['threshold']).to(device)

    # visualize the particular result specified by seed
    for file_name in os.listdir(args.res_path):
        if "seed" + str(args.seed) in file_name and file_name.endswith('pickle') and file_name.startswith('nofis'):
            data_path = os.path.join(args.res_path, file_name)
            break
    else:
        print("The stored data is not found")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print("failure rate: %.3e" % data['prob'])

    if args.loss_plot:
        plt.figure()
        plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
        for cur_step in range(len(configs['threshold']) - 1):
            cur_index = configs['nblk'] * (cur_step + 1)
            plt.plot(np.log10(data['loss_list'][cur_step]), linewidth=line_width, label=r'$q_{{{}}}$'.format(cur_index))
            ax = plt.gca()
            locator = MaxNLocator(nbins=5)
            # plt.gca().yaxis.set_major_locator(locator)
            plt.gca().xaxis.set_major_locator(locator)
            plt.yticks(size=font_size)
            plt.xticks(size=font_size)
            ax.spines['bottom'].set_linewidth(line_width)
            ax.spines['left'].set_linewidth(line_width)
            ax.spines['right'].set_linewidth(line_width)
            ax.spines['top'].set_linewidth(line_width)
            plt.legend(prop={'size': 28}, ncol=2)
            plt.title('loss curve in Log10 scale (Case %d)' % testcase_dict['case_index'])
        if args.save:
            plt.title('')
            plt.savefig(os.path.join(args.res_path, 'loss_case' + str(testcase_dict['case_index']) + '_seed' + str(args.seed) + '.png'))

    for file_name in os.listdir(args.res_path):
        if "seed" + str(args.seed) in file_name and file_name.endswith('pt'):
            model_path = os.path.join(args.res_path, file_name)
            break
    else:
        raise RuntimeError("The neural network model is not found")

    num_steps = len(configs['threshold']) - 1
    net = CTFlow(num_steps, testcase_dict['dim_x'], configs['hidden_dims'], configs['nblk'],
                 testcase_dict['log_px']).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    visual_samples = testcase_dict['sampler'](int(float(args.num_samples)))
    if testcase_dict['dim_x'] == 2:
        for i in range(1, num_steps + 1):
            visual_trans_samples, _ = net(visual_samples, i)
            plt.figure(figsize=[6, 6])
            # locator = MaxNLocator(nbins=num_ticks)
            # plt.gca().yaxis.set_major_locator(locator)
            # plt.gca().xaxis.set_major_locator(locator)
            xt = np.linspace(range_visualization[args.testcase][0][0],
                             range_visualization[args.testcase][0][1],
                             num_ticks)
            yt = np.linspace(range_visualization[args.testcase][1][0],
                             range_visualization[args.testcase][1][1],
                             num_ticks)
            plt.gca().yaxis.set_major_locator(FixedLocator(yt))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.gca().xaxis.set_major_locator(FixedLocator(xt))
            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            plt.yticks(yt, size=font_size)
            plt.xticks(xt, size=font_size)

            # ax.set_xticklabels(['%.2f' % val for val in xt])
            # ax.set_yticklabels(['%.2f' % val for val in yt])

            if i == num_steps:
                plt.hist2d(torch2np(visual_trans_samples)[:, 0], torch2np(visual_trans_samples)[:, 1], bins=200,
                           range=range_visualization[args.testcase], cmap=cmap)
            else:
                plt.hist2d(torch2np(visual_trans_samples)[:, 0], torch2np(visual_trans_samples)[:, 1], bins=200,
                           range=range_visualization[args.testcase], cmap=cmap)
            plt.subplots_adjust(left=0.15, right=0.9, top=0.9, bottom=0.15)
            ax = plt.gca()
            ax.spines['bottom'].set_linewidth(line_width)
            ax.spines['left'].set_linewidth(line_width)
            ax.spines['right'].set_linewidth(line_width)
            ax.spines['top'].set_linewidth(line_width)
            plt.title('Sampled from px and then transformed w/ %d' % i)
            if args.save:
                if i != num_steps:
                    if args.middle_plot:
                        print('saving middle plot')
                        plt.title('')
                        plt.savefig(os.path.join(args.res_path, 'learnt_proposal_' + str(i) + '_' + str(args.testcase) + '.png'), bbox_inches='tight')
                else:
                    print('saving final plot')
                    plt.title('')
                    plt.savefig(os.path.join(args.res_path, 'learnt_proposal_' + str(i) + '_' + str(args.testcase) + '.png'), bbox_inches='tight')

    if args.box_plot:
        # re-run IS here for a boxplot
        golden = goldenresult_dict[args.testcase]
        repeat = 100
        N_list = [50, 100, 200, 500]
        re_prob = np.zeros((len(N_list), repeat))
        for j in range(len(N_list)):
            cur_N = N_list[j]
            for i in range(repeat):
                samples = testcase_dict['sampler'](cur_N)
                trans_samples, log_prob = net(samples, num_steps)
                ratio = torch.exp(testcase_dict['log_px'](trans_samples) - log_prob)
                perf = testcase_dict['simulator'](trans_samples)
                index = indicator(perf, testcase_dict['thred'])
                re_prob[j, i] = torch.mean(ratio * index).item()
        plt.figure(figsize=[6, 6])
        locator = MaxNLocator(nbins=num_ticks)
        plt.gca().yaxis.set_major_locator(locator)
        boxplot = plt.boxplot(np.log10(re_prob).transpose(), showfliers=False, patch_artist=True)
        box_color = 'lightblue'
        for box in boxplot['boxes']:
            box.set(facecolor=box_color)
        # Customize the mean line color
        mean_color = 'red'
        for mean in boxplot['means']:
            mean.set(color=mean_color)

        plt.xticks([1, 2, 3, 4], [str(N) for N in N_list])
        plt.axhline(y=math.log10(golden), color='red', linewidth=3, linestyle='-', label=r'Log($P_r$)')
        plt.axhline(y=math.log10(args.margin * golden), linewidth=3, color='green', linestyle='--'
                    ,label='Log(' + '%.2f' % args.margin + r'$P_r$)')
        plt.axhline(y=math.log10(1.0 / args.margin * golden), linewidth=3, color='green', linestyle='-.'
                    ,label='Log(' + '%.2f' % (1.0 / args.margin) + r'$P_r$)')
        plt.legend(prop={'size': 28})
        plt.title('Log10 Box plot')

        plt.yticks(size=font_size)
        plt.xticks(size=font_size)
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(line_width)
        ax.spines['left'].set_linewidth(line_width)
        ax.spines['right'].set_linewidth(line_width)
        ax.spines['top'].set_linewidth(line_width)

        if args.save:
            print('Save box figure...')
            plt.title('')
            plt.savefig(os.path.join(args.res_path, 'box_' + str(args.testcase) + '.png'), bbox_inches='tight')

        if not args.not_show:
            plt.show()
