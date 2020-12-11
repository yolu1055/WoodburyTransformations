import numpy as np
import torch
import torch.nn as nn




class ChannelTrans(nn.Module):

    def __init__(self, c, d):
        super().__init__()

        self.c = c
        self.d = d

        w1 = torch.normal(mean=torch.zeros([c,d]), std=torch.ones([c,d])*0.01)
        w2 = torch.normal(mean=torch.zeros([d,c]), std=torch.ones([c,d])*0.01)
        self.register_parameter("U", nn.Parameter(torch.Tensor(w1), requires_grad=True))
        self.register_parameter("V", nn.Parameter(torch.Tensor(w2), requires_grad=True))


        self.register_parameter("eye", nn.Parameter(torch.eye(self.d), requires_grad=False))


    def forward_transform(self, x):
        h = x
        _,_,N = x.size()

        dlogdet = torch.slogdet(self.eye + torch.matmul(self.V, self.U))[1] * N

        z1 = torch.matmul(self.V, x)
        z2 = torch.matmul(self.U, z1)
        z = h + z2


        return z, dlogdet


    def inverse(self, z):
        _,_,N = z.size()
        h = z
        dlogdet = torch.slogdet(self.eye + torch.matmul(self.V, self.U))[1] * N

        B_inv = torch.inverse(self.eye + torch.matmul(self.V, self.U))

        x1 = torch.matmul(self.V, z)

        W = torch.matmul(self.U, B_inv)

        x2 = torch.matmul(W, x1)

        x = h - x2


        return x, dlogdet


    def forward(self, x, logdet=0.0, reverse=False):

        if reverse is False:
            z, dlogdet = self.forward_transform(x)
            logdet = logdet + dlogdet


        if reverse is True:
            z,dlogdet = self.inverse(x)
            logdet = logdet - dlogdet

        return z, logdet


class SpatialTrans(nn.Module):

    def __init__(self, N, d):

        super().__init__()
        self.N = N
        self.d = d

        w1 = torch.normal(mean=torch.zeros([N,d]), std=torch.ones([N,d])*0.01)
        w2 = torch.normal(mean=torch.zeros([d,N]), std=torch.ones([d,N])*0.01)
        self.register_parameter("U", nn.Parameter(torch.Tensor(w1), requires_grad=True))
        self.register_parameter("V", nn.Parameter(torch.Tensor(w2), requires_grad=True))


        self.register_parameter("eye", nn.Parameter(torch.eye(self.d), requires_grad=False))


    def forward_transform(self, x):
        _,C,_ = x.size()
        h = x

        dlogdet = torch.slogdet(self.eye + torch.matmul(self.V, self.U))[1] * C

        z1 = torch.matmul(x, self.U)
        z2 = torch.matmul(z1,self.V)
        z = h + z2

        return z, dlogdet


    def inverse(self, z):

        _,C,_ = z.size()
        h = z

        dlogdet = torch.slogdet(self.eye + torch.matmul(self.V, self.U))[1] * C

        B_inv = torch.inverse(self.eye + torch.matmul(self.V, self.U))

        z1 = torch.matmul(z, self.U)

        z2 = torch.matmul(z1, B_inv)

        z3 = torch.matmul(z2, self.V)

        x = h - z3


        return x, dlogdet


    def forward(self, x, logdet, reverse=False):

        if reverse is False:
            z, dlogdet = self.forward_transform(x)
            logdet = logdet + dlogdet


        if reverse is True:
            z, dlogdet = self.inverse(x)
            logdet = logdet - dlogdet

        return z, logdet




class WTrans(nn.Module):

    def __init__(self, C, N, d_c, d_s):

        super().__init__()

        self.CTrans = ChannelTrans(C, d_c)
        self.STrans = SpatialTrans(N, d_s)


    def forward(self, x, logdet, reverse=False):

        B,C,H,W = x.size()

        x = x.view(B,C,-1)

        if reverse is False:
            z1, logdet = self.CTrans(x, logdet, False)
            z, logdet = self.STrans(z1, logdet, False)

        if reverse is True:
            z1, logdet = self.STrans(x, logdet, True)
            z, logdet = self.CTrans(z1, logdet, True)

        z = z.view(B,C,H,W)

        return z, logdet



class MEWTrans(nn.Module):

    def __init__(self, C, H, W, d_c, d_h, d_w):

        super().__init__()

        self.CTrans = ChannelTrans(C, d_c)
        self.WTrans = SpatialTrans(W, d_w)
        self.HTrans = SpatialTrans(H, d_h)


    def forward(self, x, logdet, reverse=False):

        if reverse is False:
            B,C,H,W = x.size()
            x = x.view(B,C,H*W)
            z1, logdet = self.CTrans(x, logdet, False)
            z1 = z1.view(B,C,H,W)
            z1 = z1.view(B,C*H,W)
            z2, logdet = self.WTrans(z1, logdet, False)
            z2 = z2.view(B,C,H,W)
            z2 = z2.permute(0,1,3,2).contiguous()
            z2 = z2.view(B,C*W,H)
            z, logdet = self.HTrans(z2, logdet, False)
            z = z.view(B,C,W,H)
            z = z.permute(0,1,3,2).contiguous()
            return z, logdet

        else:
            B,C,H,W = x.size()
            x = x.permute(0,1,3,2).contiguous()
            x = x.view(B,C*W,H)
            z2, logdet = self.HTrans(x, logdet, True)
            z2 = z2.view(B,C,W,H)
            z2 = z2.permute(0,1,3,2).contiguous()
            z2 = z2.view(B,C*H,W)
            z1, logdet = self.WTrans(z2, logdet, True)
            z1 = z1.view(B,C,H,W)
            z1 = z1.view(B,C,H*W)
            z, logdet = self.CTrans(z1, logdet, True)
            z = z.view(B,C,H,W)
            return z, logdet




if __name__ == "__main__":

    B,C,H,W = 2,3,32,32
    d_c, d_s = 16,16
    d_h, d_w = 16, 16

    x_np = np.random.rand(B,C,H,W)
    x = torch.Tensor(x_np)

    import time
    start = time.time()

    trans = WTrans(C,H*W, d_c,d_s)

    z, logdet = trans(x, 0.0, False)
    print(logdet)
    x_inv, logdet = trans(z, logdet, True)
    print(logdet)
    print(torch.max(torch.abs(x-x_inv)))
    print ("elapsed time: {:.5f} ms".format(1000.0*(time.time()-start)))


    start = time.time()

    trans = MEWTrans(C,H, W, d_c,d_h, d_w)

    z, logdet = trans(x, 0.0, False)
    print(logdet)
    x_inv, logdet = trans(z, logdet, True)
    print(logdet)
    print(torch.max(torch.abs(x-x_inv)))
    print ("elapsed time: {:.5f} ms".format(1000.0*(time.time()-start)))
