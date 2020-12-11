import matplotlib
matplotlib.use("TkAgg")
import torch
from torchvision import transforms
from torchvision import datasets


def load_cifar10(path, portion):
    transform = transforms.Compose([transforms.ToTensor()])
    if portion == "train":
        dataset = datasets.CIFAR10(path, train=True, transform=transform, download=True)

    if portion == "test":
        dataset = datasets.CIFAR10(path, train=False, transform=transform, download=True)

    return dataset



def preprocess(img, n_bits=8, noise=True):
    n_bins = 2. ** n_bits
    # rescale to 255
    img = img.mul(255)
    if n_bits < 8:
        img = torch.floor(img.div(256. / n_bins))

    # normalize
    img = img.div(n_bins)
    #img = (img - 0.5).div(0.5)
    img = img - 0.5

    # add noise
    if noise == True:
        img = img + torch.zeros_like(img).uniform_(0., 1./n_bins)


    return img


def postprocess(img, n_bits=8):
    n_bins = 2. ** n_bits
    # re-normalize
    #img = img.mul(0.5) + 0.5
    img = img + 0.5
    img = img.mul(n_bins)
    # scale
    img = torch.floor(img) * (256. / n_bins)
    img = img.clamp(0, 255).div(255)
    return img
