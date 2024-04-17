from utils.testcase import get_testcase
from utils.train import train_by_reverse_kl
from utils.helper import set_seed
import torch
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--gpu', type=int, default=-1, help='device id')
    parser.add_argument('--testcase', type=int, default=5, help='the index of testbench')
    parser.add_argument('--config', default='', help='path to the config file')
    parser.add_argument('--plt_fig', default='show',
                        help='empty string: no figs, show: show figs, save: save figs to the path')
    parser.add_argument('--save_data', default=0, help='0: do not save data, 1: save data')
    parser.add_argument('--save_path', default='', help='where the figs and data will be saved to')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu") if args.gpu < 0 else torch.device("cuda:" + str(args.gpu))

    if args.config == '':
        args.config = './configs/config' + str(args.testcase) + '.json'

    testcase_dict = get_testcase(args.testcase, device)
    with open(args.config) as json_file:
        config_file = json.load(json_file)
        config_file['threshold'] = torch.Tensor(config_file['threshold']).to(device)

    prob, total_simulator_calls, *_ = train_by_reverse_kl(testcase_dict, config_file, device,
                                                          file_suffix='nofis_' + 'seed' + str(args.seed) + '_',
                                                          exp_param={'plt_fig': args.plt_fig,
                                                                     'save_data': args.save_data,
                                                                     'save_path': args.save_path})

    print('Proposed:  estimate = %.3e, #sample = %d' % (prob, total_simulator_calls))

    '''
    goldenresult_dict = { 101: 4.688e-04, # w/ 5e9 samples run on server, June 29, 2023
                      102: 3.656e-6, # w/ 5e9 samples run on server, June 29, 2023
                      103: 4.74e-6, # w/ 5e9 samples run on server, June 28, 2023
                      105: 2.1516e-9, # analytical [1 - normcdf(1.8)] ^ 6
                      106: 3.149e-05,
                     }
    '''