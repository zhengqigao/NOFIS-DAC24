import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class Feedforward(nn.Module):
    def __init__(self, hidden_dims, activation='leakyrelu'):
        super(Feedforward, self).__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)])

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x):

        for i in range(len(self.hidden_layers) - 1):
            x = self.activation(self.hidden_layers[i](x))
        x = self.hidden_layers[-1](x)
        return x


class Transformation(nn.Module):
    def __init__(self, dim, hidden_dims, parity, scale, shift, activation):
        super(Transformation, self).__init__()
        self.hidden_dims, self.dim = hidden_dims, dim
        self.scale, self.shift = scale, shift

        self.parity = parity
        if scale:
            self.snet = Feedforward([max(1, self.dim // 2), *hidden_dims, max(1, self.dim // 2)], activation)
        else:
            self.snet = lambda x: x.new_zeros(x.size(0), self.dim // 2)

        if shift:
            self.tnet = Feedforward([max(1, self.dim // 2), *hidden_dims, max(1, self.dim // 2)], activation)
        else:
            self.tnet = lambda x: x.new_zeros(x.size(0), self.dim // 2)

        assert dim > 1  # Note that the architecture of RealNVP fails when dim=1 (the log_det is wrong)

    def forward(self, x, alpha=0.0):
        x0, x1 = x[:, 0::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.snet(x0)
        t = self.tnet(x0)
        z0, z1 = x0, (1 - alpha) * (torch.exp(s) * x1 + t) + alpha * x1
        if self.parity:
            z0, z1 = z1, z0
        z = torch.stack((z0, z1), dim=1).transpose(1, 2).reshape(z0.size(0), -1)  # z=[z0[0],z1[0],z0[1],z1[1],...]
        log_det = torch.log(torch.prod((1 - alpha) * torch.exp(s) + alpha, dim=1))
        return z, log_det

    def backward(self, z, alpha=0.0):
        z0, z1 = z[:, 0::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.snet(z0)
        t = self.tnet(z0)
        x0, x1 = z0, torch.div((z1 - (1 - alpha) * t), (1 - alpha) * torch.exp(s) + alpha)
        if self.parity:
            x0, x1 = x1, x0
        x = torch.stack((x0, x1), dim=1).transpose(1, 2).reshape(x0.size(0), -1)  # x=[x0[0],x1[0],x0[1],x1[1],...]
        log_det = -torch.log(torch.prod((1 - alpha) * torch.exp(s) + alpha, dim=1))
        return x, log_det  # forward and backward log_det sum to zero


class TransformationBlock(nn.Module):
    def __init__(self, dim, hidden_dims, nblk):
        super(TransformationBlock, self).__init__()

        self.flows = nn.ModuleList(
            [Transformation(dim, hidden_dims, i % 2 == 1, scale=True, shift=True, activation='leakyrelu') for i in
             range(nblk)])
        assert dim > 1  # Note that the architecture of RealNVP fails when dim=1 (the log_det is wrong)

    def forward(self, x, alpha=0.0):
        log_det = 0
        for i in range(len(self.flows)):
            x, ld = self.flows[i].forward(x, alpha)
            log_det += ld
        return x, log_det

    def backward(self, z, alpha=0.0):
        log_det = 0
        for i in range(len(self.flows)):
            z, ld = self.flows[-1 - i].backward(z, alpha)
            log_det += ld
        return z, log_det  # forward and backward log_det sum to zero


class CTFlow(nn.Module):
    def __init__(self, num_steps, dim, hidden_dims, nblk, log_p0):
        super(CTFlow, self).__init__()
        self.transforms = nn.ModuleList([TransformationBlock(dim, hidden_dims, nblk) for _ in range(num_steps)])
        self.log_p0 = log_p0
        self.num_steps = num_steps
        assert dim > 1  # Note that the architecture of RealNVP fails when dim=1 (the log_det is wrong)

    def forward(self, x, at_step=-1):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        log_prob = self.log_p0(x)
        for i in range(self.num_steps if at_step == -1 else at_step):
            x, ld = self.transforms[i].forward(x, alpha=0.0)
            log_prob -= ld
        return x, log_prob

    def backward(self, z):
        if z.ndim == 1:
            z = z.reshape(1, -1)

        log_prob = 0
        for i in range(len(self.transforms)):
            z, ld = self.transforms[-1 - i].backward(z, alpha=0.0)
            log_prob += ld
        return z, log_prob + self.log_p0(z) # forward and backward log_det equal

    def freeze(self, till_step):
        for i in range(till_step):
            for param in self.transforms[i].parameters():
                param.requires_grad = False


if __name__ == '__main__':
    alpha = 0.0
    dim = 2
    net = TransformationBlock(dim, [4, 4], 1)
    x = torch.rand(10, dim)
    z, log = net(x, alpha)
    x0, log2 = net.backward(z, alpha)
    print("diff recover: ", torch.max(torch.abs(x - x0)))
    print("diff log_det:", torch.max(torch.abs(log + log2)))

    mean, cov = torch.zeros(dim), torch.diag(torch.ones(dim))
    mn_function = MultivariateNormal(mean, cov)
    init_pdf = lambda x: torch.exp(mn_function.log_prob(x))
    num_steps = 4
    net = CTFlow(num_steps, dim, [4, 4], 2, init_pdf)
    z, log_det = net(x, alpha * torch.ones(num_steps, ))
    x0, log_det2 = net.backward(z, alpha * torch.ones(num_steps, ))
    print("diff recover: ", torch.max(torch.abs(x - x0)))
    print("diff log_det:", torch.max(torch.abs(log + log2)))
