import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modules
import WoodburyTrans




class WoodburyGlowStep(nn.Module):

    def __init__(self, C, H, W, d, is_me, hidden_channels):

        super().__init__()

        # 1. actnorm
        self.actnorm = modules.Actnorm(C)

        # 2. conv
        self.conv = None
        if not is_me:
            assert len(d) >= 2, ("Woodbury layer has 2 transformations")
            self.conv = WoodburyTrans.WTrans(C,H*W,d[0],d[1])

        else:
            assert len(d) >= 3, ("ME-Woodbury layer has 3 transformations")
            self.conv = WoodburyTrans.MEWTrans(C,H,W,d[0],d[1],d[2])


        # 3. affine
        self.affine = modules.AffineCoupling(C, hidden_channels)



    def forward(self, input, logdet=0.0, reverse=False):

        if reverse is False:
            return self.normal_flow(input, logdet)
        if reverse is True:
            return self.reverse_flow(input, logdet)

    def normal_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        #1 actnorm
        z, logdet = self.actnorm(input, logdet, reverse=False)

        #2 conv
        #z, logdet = self.conv(input, logdet, reverse=False)
        z, logdet = self.conv(z, logdet, reverse=False)

        #3 affine coupling
        z, logdet = self.affine(z, logdet, reverse=False)

        return z, logdet


    def reverse_flow(self, input, logdet):
        assert input.size(1) % 2 == 0

        #1 affine coupling
        z, logdet = self.affine(input, logdet, reverse=True)

        #2 conv
        #z, logdet = self.conv(input, logdet, reverse=True)
        z, logdet = self.conv(z, logdet, reverse=True)

        #3 actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)

        return z, logdet



class FlowNet(nn.Module):

    def __init__(self, image_shape, hidden_channels, K, L, d, is_me):
        """
                             K                                      K
        --> [Squeeze] -> [FlowStep] -> [Split] -> [Squeeze] -> [FlowStep]
               ^                           v
               |          (L - 1)          |
               + --------------------------+
        """

        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        self.L = L
        C,H,W = image_shape
        assert C == 1 or C == 3, ("image_shape should be HWC, like (64, 64, 3)"
                                  "C == 1 or C == 3")


        if is_me:
            assert len(d) >= 3, ("ME-Woodbury layer has 3 transformations")

        else:
            assert len(d) >= 2, ("Woodbury layer has 2 transformations")


        assert len(d[0]) >= L, ("need to specify latent dimension for each level")

        # small
        # d_c = [8,8,16]
        # d_s = [16,8,8]

        d_c = [16,16,16]
        d_s = [16,16,16]

        # 256x256 L=6
        # d_c = [8,8,16,16,16,16]
        # d_s = [16,16,16,16,8,8]

        # 128x128 L=5
        # d_c = [8,8,16,16,16]
        # d_s = [16,16,16,8,8]

        # 64x64 L=4
        # d_c = [8,8,16,16]
        # d_s = [16,16,8,8]
        # d_c = [32,32,16,16]
        # d_s = [16,16,32,32]


        # big
        # d_c = [16, 32, 64]
        # d_s = [128, 64, 32]

        # same
        # d_c = [16, 16, 16]
        # d_s = [16, 16, 16]

        # d_c = [32, 32, 32]
        # d_s = [32, 32, 32]
        # d_c = [8, 8, 8]
        # d_s = [8, 8, 8]
        # d_c = [16, 8, 8]
        # d_s = [16, 8, 8]


        for l in range(0, L):

            # 1. Squeeze
            C, H, W = C * 4, H // 2, W // 2
            self.layers.append(modules.SqueezeLayer(2))
            self.output_shapes.append([-1, C, H, W])

            # 2. K FlowStep
            for k in range(0, K):
                if is_me:
                    self.layers.append(WoodburyGlowStep(C, H, W, [d[0][l], d[1][l], d[2][l]], is_me, hidden_channels))
                else:
                    self.layers.append(WoodburyGlowStep(C, H, W, [d[0][l], d[1][l]], is_me, hidden_channels))

                self.output_shapes.append([-1, C, H, W])

            # 3. Split2d
            if l < L - 1:
                self.layers.append(modules.Split(num_channels=C))
                self.output_shapes.append([-1, C // 2, H, W])
                C = C // 2


    def forward(self, input, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            return self.encode(input, logdet)
        else:
            return self.decode(input, eps_std)

    def encode(self, z, logdet=0.0):
        for layer, shape in zip(self.layers, self.output_shapes):
            z, logdet = layer(z, logdet, reverse=False)
        return z, logdet

    def decode(self, z, eps_std=None):
        for layer in reversed(self.layers):
            if isinstance(layer, modules.Split):
                z, logdet = layer(z, logdet=0.0, reverse=True, eps_std=eps_std)
            else:
                z, logdet = layer(z, logdet=0.0, reverse=True)
        return z, logdet



class WoodburyGlow(nn.Module):
    BCE = nn.BCEWithLogitsLoss()
    CE = nn.CrossEntropyLoss()

    def __init__(self, args):
        super().__init__()
        self.flow = FlowNet(image_shape=args.image_shape,
                            hidden_channels=args.hidden_channels,
                            K=args.flow_depth,
                            L=args.num_levels,
                            d=args.ranks,
                            is_me=args.is_me)

        self.learn_top = args.learn_top

        # prior
        self.register_parameter("latent_mean",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))


        self.register_parameter("latent_logs",
                                nn.Parameter(torch.zeros(
                                    [1,
                                     self.flow.output_shapes[-1][1],
                                     self.flow.output_shapes[-1][2],
                                     self.flow.output_shapes[-1][3]])))

        self.n_bins = float(2.**args.n_bits)


    def prior(self):

        if self.learn_top:
            return self.latent_mean, self.latent_logs
        else:
            return torch.zeros_like(self.latent_mean), torch.zeros_like(self.latent_mean)


    def forward(self, x=None, z=None, eps_std=0.7, reverse=False):
        if reverse == False:
            pixels = x.size(1) * x.size(2) * x.size(3)
            logdet = torch.zeros_like(x[:, 0, 0, 0])
            logdet += float(-np.log(self.n_bins) * pixels) # Equation 2, discretization level of data
            # encode
            z, objective = self.flow(x, logdet=logdet, reverse=False)
            # prior
            mean, logs = self.prior()
            objective += modules.GaussianDiag.logp(mean, logs, z)
            #nll = (-objective / pixels)
            nll = -objective / float(np.log(2.) * pixels)
            return z, nll
        else:
            with torch.no_grad():
                mean, logs = self.prior()
            if z is None:
                z = modules.GaussianDiag.sample(mean, logs, eps_std)
            x, logdet = self.flow(z, eps_std=eps_std, reverse=True)
            return x, logdet
