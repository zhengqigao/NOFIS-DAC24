import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import math


def compute_U1(z, bias=2.0):
    mask = torch.ones_like(z)
    mask[:, 1] = 0.0

    U = (0.5 * ((torch.norm(z, dim=-1) - bias) / 0.4) ** 2 - \
         torch.sum(mask * torch.log(torch.exp(-0.5 * ((z - bias) / 0.6) ** 2) +
                                    torch.exp(-0.5 * ((z + bias) / 0.6) ** 2)), -1))
    return U


def compute_U3(z, bias=0.0):
    z = z + bias
    w1 = torch.sin(2 * math.pi * z[:, 0] / 4)
    w2 = 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
    U = -torch.log(
        torch.exp(-0.5 * ((z[:, 1] - w1) / 0.35) ** 2) + torch.exp(-0.5 * ((z[:, 1] - w1 + w2) / 0.35) ** 2) + math.exp(
            -0.01))
    return U


def customized_heart(x, y):
    bias = 3
    t = torch.atan2(y - bias, x - bias)  # convert to polar coordinates
    r = torch.sqrt((x - bias) ** 2 + (y - bias) ** 2)
    res = (16 * torch.sin(t) ** 3 * (torch.sin(t) ** 2 * torch.cos(t) ** 2 + torch.cos(t) ** 2) ** (-1 / 3) +
           13 * torch.cos(t) - 5 * torch.cos(2 * t) - 2 * torch.cos(3 * t) - torch.cos(4 * t)) / r
    return res


def customized_potential(x):
    bias = 3.0
    w1 = torch.sin(2 * np.pi * (x[:, 0] - bias) / 4.0)
    w2 = 3.0 * torch.exp(-0.5 * ((x[:, 0] - 1 - bias) / 0.6) ** 2)
    tmp1 = (x[:, 1] - bias - w1) / 0.35
    tmp2 = (x[:, 1] - bias - w1 + w2) / 0.35
    y = -torch.log(torch.exp(-0.5 * tmp1 ** 2) + torch.exp(-0.5 * tmp2 ** 2) + np.exp(-5))

    return y


def customized_ring(x):
    t = torch.atan2(x[:, 1], x[:, 0])
    y = (x[:, 0] - 0) ** 2 + (x[:, 1] - 0) ** 2
    # y[x[:, 0] <= 0] = 30
    # y[x[:, 1] <= 0] = 30
    # y[(t >= 0.4 * np.pi) | (t <= 0.2 * np.pi)] = 30
    return y


def rosenbrock(x, bias=0.0):
    '''
    Rosenbrock function: https://en.wikipedia.org/wiki/Rosenbrock_function

    :param x: (N,D) tensor
    :return: (N,) tensor

    '''
    t2 = torch.sum((1 - (x[:, :-1] - bias)) ** 2, dim=1)
    t1 = torch.sum(100 * ((x[:, :-1] - bias) ** 2 - (x[:, 1:] - bias)) ** 2, dim=1)
    y = t1 + t2
    return 0.01 * y


def levy(x, bias=0.0):
    '''
    Levy function: https://www.sfu.ca/~ssurjano/levy.html

    :param x: (N,D) tensor
    :return: (N,) tensor

    '''
    x = x - bias
    w = 1 + (x - 1) / 4.0
    t1 = torch.sin(math.pi * w[:, 0]) ** 2
    t2 = (w[:, :-1] - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w[:, :-1] + 1) ** 2)
    t3 = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * math.pi * w[:, -1]))
    y = t1 + torch.sum(t2, dim=1) + t3

    return y


def rastrigin(x):
    '''
    Rastrigin function: https://www.sfu.ca/~ssurjano/rastr.html

    :param x: (N,D) tensor
    :return: (N,) tensor
    '''
    bias = torch.ones_like(x) * 0.0
    y = torch.sum((x - bias) ** 2 - 10 * torch.cos(2 * math.pi * (x - bias)), dim=1) + 10 * x.size(1)
    return 0.01 * y


def powell(x):
    '''
    Powell function: https://www.sfu.ca/~ssurjano/powell.html

    :param x: (N,D) tensor
    :return: (N,) tensor
    '''
    if x.size(1) < 4:
        raise ValueError("In Powell function, the dimension of design varaible must be no smaller than 4")
    dim_end = int(x.size(1) / 4) * 4
    t1 = (x[:, 0:dim_end:4] + 10 * x[:, 1:dim_end:4]) ** 2
    t2 = 5 * (x[:, 2:dim_end:4] - x[:, 3:dim_end:4]) ** 2
    t3 = (x[:, 1:dim_end:4] - 2 * x[:, 2:dim_end:4]) ** 4
    t4 = 10 * (x[:, 0:dim_end:4] - x[:, 3:dim_end:4]) ** 4
    y = torch.sum(t1 + t2 + t3 + t4, dim=1)
    return 0.01 * y


def get_testcase(index, device=torch.device("cpu")):
    if index == 1:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [0.001, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: compute_U1(x).reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 2:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [0, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: torch.min((x[:, 0] + 3.8) ** 2 + (x[:, 1] + 3.8) ** 2 - 1,
                                        (x[:, 0] - 3.8) ** 2 + (x[:, 1] - 3.8) ** 2 - 1).reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 3:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-1e15, ], [0.001, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: customized_potential(x).reshape(-1, dim_y)
        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 4:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [-0.6, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: compute_U3(x, bias=2.4).reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    # 2D Gaussian, integral region is an ellipsoid
    elif index == 5:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[16, ], [20.25, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: customized_ring(x).reshape(-1, dim_y)
        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    # 2D Gaussian, integral at a heart shape
    elif index == 6:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-50, ], [-40, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: customized_heart(x[:, 0], x[:, 1]).reshape(-1, dim_y)
        log_px = lambda x: torch.exp(mn_function.log_prob(x))
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 101:
        dim_x = 10
        dim_y = 1
        thred = torch.Tensor([[3.48, ], [3.52, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: rosenbrock(x).reshape(-1, 1)
        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 102:
        dim_x = 20
        dim_y = 1
        thred = torch.Tensor([[-0, ], [6, ]]).to(device)  # 0, 6
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: levy(x, 1.0).reshape(-1, 1)
        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))
    elif index == 103:
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [0, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: torch.min((x[:, 0] + 3.8) ** 2 + (x[:, 1] + 3.8) ** 2 - 1,
                                        (x[:, 0] - 3.8) ** 2 + (x[:, 1] - 3.8) ** 2 - 1).reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))
    elif index == 104:  # We know the analytical golden result (integral of Gaussian)
        dim_x = 2
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [0, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: torch.max(4 - x[:, 0], 4 - x[:, 1]).reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 105:  # We know the analytical golden result (integral of Gaussian)
        dim_x = 6
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [0, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: torch.max(1.8 - x, dim=1)[0].reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    elif index == 106:
        dim_x = 40
        dim_y = 1
        thred = torch.Tensor([[-np.inf, ], [4, ]]).to(device)
        mean, cov = torch.zeros(dim_x).to(device), torch.diag(torch.ones(dim_x)).to(device)
        mn_function = MultivariateNormal(mean, cov)
        simulator = lambda x: powell(x).reshape(-1, dim_y)

        log_px = lambda x: mn_function.log_prob(x)
        sampler = lambda num_sample: mn_function.sample(sample_shape=(num_sample,))

    else:
        raise ValueError("Test case is not implemented")

    return {'dim_x': dim_x, 'dim_y': dim_y, 'thred': thred, 'simulator': simulator,
            'log_px': log_px, 'sampler': sampler, 'case_index': index}
