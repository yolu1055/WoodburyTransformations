import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.linalg as la

logabs = lambda x: torch.log(torch.abs(x))

class Actnorm(nn.Module):

    def __init__(self, num_channels):

        super().__init__()
        size = [1, num_channels, 1, 1]
        bias = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        logs = torch.normal(mean=torch.zeros(*size), std=torch.ones(*size)*0.05)
        self.register_parameter("bias", nn.Parameter(torch.Tensor(bias)))
        self.register_parameter("logs", nn.Parameter(torch.Tensor(logs)))
        self.bias.data.copy_(bias.data)
        self.logs.data.copy_(logs.data)
        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))


    def forward(self, input, logdet=0.0, reverse=False):
        dimentions = input.size(2) * input.size(3)
        if self.initialized.item() == 0:
            self.data_based_initialization(input)

        if reverse == False:

            input = input + self.bias
            input = input * torch.exp(self.logs)
            dlogdet = torch.sum(self.logs) * dimentions
            logdet = logdet + dlogdet

        if reverse == True:
            input = input * torch.exp(-self.logs)
            input = input - self.bias
            dlogdet = - torch.sum(self.logs) * dimentions
            logdet = logdet + dlogdet

        return input, logdet


    def data_based_initialization(self, input):
        with torch.no_grad():
            bias = torch.mean(input.clone(), dim=[0, 2, 3], keepdim=True) * -1.0
            vars = torch.mean((input.clone() + bias) ** 2, dim=[0, 2, 3], keepdim=True)
            logs = torch.log(1.0/(torch.sqrt(vars)+1e-6))
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.initialized.fill_(1)




class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 pad="same", do_actnorm=True, weight_std=0.05):
        padding = Conv2d.get_padding(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, bias=(not do_actnorm))
        #initialize weight
        self.weight.data.normal_(mean=0.0, std=weight_std)

        if not do_actnorm:
            self.bias.data.zero_()

        else:
            self.actnorm = Actnorm(out_channels)

        self.do_actnorm = do_actnorm

    def forward(self, input):
        x = super().forward(input)
        if self.do_actnorm:
            x, _ = self.actnorm(x)
        return x


    @staticmethod
    def get_padding(kernel_size):
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        padding = list()
        for k in kernel_size:
            padding.append((k-1)//2)

        return padding


class Conv2dZeros(nn.Conv2d):

    def __init__(self, in_channels, out_channels,
                 kernel_size=[3, 3], stride=[1, 1],
                 padding="same", logscale_factor=1.0):

        padding = Conv2d.get_padding(kernel_size)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        #logscale_factor
        self.logscale_factor = logscale_factor
        self.register_parameter("logs", nn.Parameter(torch.zeros(out_channels, 1, 1)))
        self.register_parameter("newbias", nn.Parameter(torch.zeros(out_channels, 1, 1)))

        # init
        self.weight.data.zero_()
        self.bias.data.zero_()


    def forward(self, input):
        output = super().forward(input)
        output = output + self.newbias
        output = output * torch.exp(self.logs * self.logscale_factor)
        return output



class AffineCoupling(nn.Module):

    def __init__(self, num_channels, hidden_channels):
        super().__init__()

        self.f = nn.Sequential(
            Conv2d(num_channels//2, hidden_channels),
            nn.ReLU(),
            Conv2d(hidden_channels, hidden_channels, kernel_size=[1, 1]),
            nn.ReLU(),
            Conv2dZeros(hidden_channels, num_channels)
        )


    def forward(self, x, logdet=0.0, reverse=False):
        C = x.size(1)
        z1, z2 = x[:, :C // 2, ...], x[:, C // 2:, ...]
        h = self.f(z1)

        shift, scale = h[:, 0::2, ...], h[:, 1::2, ...]
        scale = torch.sigmoid(scale + 2.)
        if reverse == False:
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = torch.sum(torch.log(scale), dim=(1, 2, 3)) + logdet
        if reverse == True:
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -torch.sum(torch.log(scale), dim=(1, 2, 3)) + logdet


        z = torch.cat((z1, z2), dim=1)

        return z, logdet



class SqueezeLayer(nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

        #self.layer_name = "SqueezeLayer"

    def forward(self, input, logdet=None, reverse=False):
        if not reverse:
            output = SqueezeLayer.squeeze2d(input, self.factor)
            return output, logdet
        else:
            output = SqueezeLayer.unsqueeze2d(input, self.factor)
            return output, logdet


    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        B,C,H,W = input.size()
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * factor * factor, H // factor, W // factor)
        return x


    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        B,C,H,W = input.size()
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, C // factor2, factor, factor, H, W)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x



class GaussianDiag:
    Log2PI = float(np.log(2 * np.pi))

    @staticmethod
    def likelihood(mean, logs, x):
        """
        lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
              k = 1 (Independent)
              Var = logs ** 2
        """
        return -0.5 * (logs * 2. + ((x - mean) ** 2.) / torch.exp(logs * 2.) + GaussianDiag.Log2PI)

    @staticmethod
    def logp(mean, logs, x):
        likelihood = GaussianDiag.likelihood(mean, logs, x)
        return torch.sum(likelihood, dim=(1, 2, 3))

    @staticmethod
    def sample(mean, logs, eps_std=None):
        eps_std = eps_std or 1
        # eps = torch.normal(mean=torch.zeros_like(mean),
        #                    std=torch.ones_like(logs) * eps_std)

        eps = torch.normal(mean=torch.zeros_like(mean),
                           std=torch.ones_like(logs))
        eps = eps * eps_std

        #return mean + torch.exp(2. * logs) * eps
        return mean + torch.exp(logs) * eps

    @staticmethod
    def batchsample(batchsize, mean, logs, eps_std=None):
        sample = None
        for i in range(0, batchsize):
            s = GaussianDiag.sample(mean, logs, eps_std)
            if sample is None:
                sample = s
            else:
                torch.cat((sample, s), dim=0)
        return sample


class Split(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.conv = Conv2dZeros(num_channels // 2, num_channels)

    def split2d_prior(self, z):
        h = self.conv(z)
        mean, logs = h[:, 0::2, ...], h[:, 1::2, ...]
        return mean, logs

    def forward(self, input, logdet=0.0, reverse=False, eps_std=None):
        if reverse is False:
            C = input.size(1)
            z1, z2 = input[:, :C // 2, ...], input[:, C // 2:, ...]
            mean, logs = self.split2d_prior(z1) # This step does not make sense. Why we use convolutional layer to generate mean and variance
            logdet = GaussianDiag.logp(mean, logs, z2) + logdet
            return z1, logdet
        if reverse is True:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = GaussianDiag.sample(mean, logs, eps_std)
            z = torch.cat((z1, z2),dim=1)
            return z, logdet
