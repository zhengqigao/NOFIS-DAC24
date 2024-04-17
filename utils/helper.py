import numpy as np
import torch
import random
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.uniform import Uniform

np2torch = lambda input, device: torch.from_numpy(input).type(torch.FloatTensor).to(device)
torch2np = lambda input: input.cpu().detach().numpy()


def indicator(y, threshold):
    '''

    :param y: (N,dim_y) tensor
    :param threshold: (2, dim_y) tensor
    :return: (N,) bool tensor
    '''

    index1 = y >= threshold[0, :]
    index2 = y <= threshold[1, :]
    res = index1 & index2
    return res.all(dim=1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


