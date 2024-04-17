import numpy as np
import torch
import torch.nn as nn
from .model import CTFlow
from .helper import indicator, torch2np
import matplotlib.pyplot as plt
import pickle
import torch.optim.lr_scheduler as lr_scheduler
import os

def log_potential(x, thred, simulator, log_nx, temp=1.0):
    y = simulator(x)
    if torch.isnan(y).sum() >= 1 or torch.isinf(y).sum() >= 1:
        raise ValueError("Simulator evaluation has nan or inf")
    log_gx = torch.clamp(temp * torch.min(thred[1] - y, y - thred[0]), max=0)
    log_potential = log_gx + log_nx(x).reshape(-1, 1)
    return log_potential


def train_by_reverse_kl(testcase_dict, configs, device, file_suffix='', exp_param={'plt_fig': '',
                                                                                   'save_data': 0,
                                                                                   'save_path': ''}):
    filename_key = file_suffix + 'case' + str(testcase_dict['case_index']) + '_' \
                   + 'b' + str(configs['batch_num']) + '_' \
                   + 'i' + str(configs['epoch']) + '_' \
                   + 'l' + str(configs['learning_rate']) + '_' \
                   + 'e' + str(configs['eval_num']) + '_' \
                   + 't' + str(configs['temperature'])

    num_steps = len(configs['threshold']) - 1
    net = CTFlow(num_steps, testcase_dict['dim_x'], configs['hidden_dims'], configs['nblk'],
                 testcase_dict['log_px']).to(device)

    loss_list = [[] for _ in range(num_steps)]

    for cur_step in range(1, num_steps + 1):

        if configs['freeze'] == 1:
            net.freeze(cur_step - 1)

        optimizer = torch.optim.Adam(net.parameters(), lr=configs['learning_rate'])

        if configs['scheduler'] == 'steplr':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
        elif configs['scheduler'] == 'none':
            scheduler = None
        else:
            raise NotImplementedError("Learning rate scheduler is not implemented")

        for iter in range(configs['epoch']):

            x = testcase_dict['sampler'](configs['batch_num'])
            trans_x, log_qx = net(x, cur_step)
            log_det = log_qx - testcase_dict['log_px'](x)
            logp = log_potential(trans_x, configs['threshold'][cur_step], testcase_dict['simulator'],
                                 testcase_dict['log_px'], configs['temperature'])
            loss = torch.mean(- logp + log_det)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_list[cur_step - 1].append(loss.item())

            if scheduler:
                scheduler.step()

        if testcase_dict['dim_x'] == 2 and exp_param['plt_fig']:
            sample_x = testcase_dict['sampler'](10000)
            plt.figure()
            sample_trans_x = torch2np(net(sample_x, cur_step)[0])
            plt.hist2d(sample_trans_x[:, 0], sample_trans_x[:, 1], bins=50)
            plt.title("step = %d" % cur_step)
            plt.savefig(os.path.join(exp_param['save_path'],  'hist' + filename_key + '_' + 'step_' + str(
                cur_step) + '.jpg')) if exp_param['plt_fig'] == 'save' else None

    if exp_param['plt_fig']:
        plt.figure()
        for cur_step in range(num_steps):
            plt.plot(loss_list[cur_step], label='step = %d' % cur_step)
            plt.legend()
            plt.title('loss curve (Case %d)' % testcase_dict['case_index'])
        plt.savefig(os.path.join(exp_param['save_path'], 'loss_' + filename_key + '.jpg')) if exp_param[
                                                                                      'plt_fig'] == 'save' else None

    if exp_param['plt_fig'] == 'show':
        plt.show()

    samples = testcase_dict['sampler'](configs['eval_num'])
    trans_samples, log_prob = net(samples, num_steps)
    ratio = torch.exp(testcase_dict['log_px'](trans_samples) - log_prob)
    perf = testcase_dict['simulator'](trans_samples)
    index = indicator(perf, testcase_dict['thred'])
    prob = torch.mean(ratio * index).item()

    total_simulator_calls = num_steps * configs['batch_num'] * configs['epoch'] + configs['eval_num']
    if exp_param['save_data']:
        d = {'prob': prob, 'num_sample': total_simulator_calls, 'loss_list': loss_list}
        with open(os.path.join(exp_param['save_path'], filename_key + '.pickle'), 'wb') as f:
            pickle.dump(d, f)
        torch.save(net.state_dict(), os.path.join(exp_param['save_path'], filename_key + '.pt'))

    return prob, total_simulator_calls, loss_list, net
