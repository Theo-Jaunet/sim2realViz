import argparse
import base64
import io
import math
import numbers

import numpy
import numpy as np
import torch
from PIL import Image

import torch.nn.functional as F
from torch import nn


def how_far(x, y):
    xn = (x.detach().cpu().numpy()[0][0:2] * 22) - 2
    yn = (y.numpy()[0][0:2] * 22) - 2

    return abs(np.linalg.norm(xn[0:2] - yn[0:2]))


def calc_angl(x, y):
    angle = math.acos(x) if y > 0 else -math.acos(x)

    if angle < 0:
        return abs(360 - abs(angle))
    else:
        return angle


def img_to_b64(img):
    saver = io.BytesIO()
    img = img.squeeze(0)

    img = img[0:3].transpose(2, 0).transpose(1, 0)
    img = ((img * 0.5) + 0.5)
    img = img.numpy()

    img = np.clip((img * 255), 0.0, 255.0).astype(np.uint8)

    Image.fromarray(img).save(saver, format="JPEG")

    return str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", "")


def simple_img_to_b64(img):
    saver = io.BytesIO()
    # img = img.squeeze(0)

    # img = img[0:3].transpose(2, 0).transpose(1, 0)
    # img = ((img * 0.5) + 0.5)
    img = img.numpy()

    img = np.clip((img * 255), 0.0, 255.0).astype(np.uint8)

    Image.fromarray(img).save(saver, format="JPEG")

    return str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", "")

def depth_to_b64(img):
    saver = io.BytesIO()
    img = img.squeeze()
    img = ((img * 0.5) + 0.5)
    img = img.numpy()

    img = np.clip((img * 255), 0.0, 255.0).astype(np.uint8)

    Image.fromarray(img, mode="L").save(saver, format="JPEG")

    return str(base64.b64encode(saver.getvalue())).replace("b'", "").replace("'", "")


def get_label(name, n_out):
    labels = name.replace('.jpeg', '').split('/')[-1].split('_')
    for i, labe in enumerate(labels[:-1]):
        if i < 3:
            labels[i] = float(labe[1:])
            # labels[i] = (float(labe[1:]) + 2) * (1 / 22)
        else:
            labels[i] = (float(labe[1:]))
    if n_out == 2:
        return torch.FloatTensor([labels[0]] + [labels[2]])
    elif n_out == 3:

        return torch.FloatTensor([labels[0]] + labels[2:4])


class CascadeGaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing separately for each channel (depthwise convolution).
    Arguments:
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """

    def __init__(self, kernel_size, sigma):
        super().__init__()

        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size, kernel_size]

        cascade_coefficients = [0.5, 1.0, 2.0]  # std multipliers
        sigmas = [[coeff * sigma, coeff * sigma] for coeff in cascade_coefficients]  # isotropic Gaussian

        self.pad = int(kernel_size[0] / 2)  # assure we have the same spatial resolution

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernels = []
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for sigma in sigmas:
            kernel = torch.ones_like(meshgrids[0])
            for size_1d, std_1d, grid in zip(kernel_size, sigma, meshgrids):
                mean = (size_1d - 1) / 2
                kernel *= 1 / (std_1d * math.sqrt(2 * math.pi)) * torch.exp(-((grid - mean) / std_1d) ** 2 / 2)
            kernels.append(kernel)

        gaussian_kernels = []
        for kernel in kernels:
            # Normalize - make sure sum of values in gaussian kernel equals 1.
            kernel = kernel / torch.sum(kernel)
            # Reshape to depthwise convolutional weight
            kernel = kernel.view(1, 1, *kernel.shape)
            kernel = kernel.repeat(4, 1, 1, 1)
            kernel = kernel.to("cuda:0")

            gaussian_kernels.append(kernel)

        self.weight1 = gaussian_kernels[0]
        self.weight2 = gaussian_kernels[1]
        self.weight3 = gaussian_kernels[2]
        self.conv = F.conv2d

    def forward(self, input):
        # print(input.size())
        input = F.pad(input, [self.pad, self.pad, self.pad, self.pad], mode='reflect')

        # Apply Gaussian kernels depthwise over the input (hence groups equals the number of input channels)
        # shape = (1, 3, H, W) -> (1, 3, H, W)
        num_in_channels = input.shape[1]
        grad1 = self.conv(input, weight=self.weight1, groups=num_in_channels)
        grad2 = self.conv(input, weight=self.weight2, groups=num_in_channels)
        grad3 = self.conv(input, weight=self.weight3, groups=num_in_channels)

        return (grad1 + grad2 + grad3) / 3


def adjust_angle(ang):
    if ang < 0:
        return ang + 360
    return ang


def fullDist(coords1, coords2, ang1, ang2):
    ratio = 1 / 22

    # coords1 *= ratio
    # coords2 *= ratio

    pos = F.mse_loss(torch.from_numpy(numpy.array(coords1)) * ratio, torch.from_numpy(numpy.array(coords2)) * ratio)

    # return (1 - ((0.6 * pos + 0.4 * angle_dist(ang1, ang2)) * 0.5)).clamp(0, 1).item()
    return 1 - pos.item()


def euclidean(a, b):
    ratio = 1 / 26
    a = numpy.array(a)
    b = numpy.array(b)

    dist = 1 - numpy.linalg.norm(a - b) * ratio

    if dist < 0:
        return 0
    else:
        return dist


def angle_dist(a, b):
    dist = (a - b + 360) % 360
    if dist > 180:
        dist = 360 - dist
    return dist * (1 / 180)


def loadsave(model, modelpath):
    model.load_state_dict(torch.load(modelpath, map_location='cuda:0'))
    return model


def parsearg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.007, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--img_height', type=int, default=256, help='size of image height')
    parser.add_argument('--img_width', type=int, default=256, help='size of image width')
    parser.add_argument('--checkpoint_interval', type=int, default=50, help='interval between model checkpoints')
    parser.add_argument('--num_workers', type=int, default=2, help='Workers for the dataset')
    parser.add_argument('--dataset_size', type=int, default=10000, help='Size of the dataset, -1 for full size')
    parser.add_argument('--n_out', type=int, default=3, help='Number of output')
    parser.add_argument('--model', type=str, default='posest', help=' Kind of model to use')

    return parser.parse_args()


def get_label(name, n_out):
    # print(name)
    labels = name.replace('.jpeg', '').split('/')[-1].split('_')
    for i, labe in enumerate(labels[:-1]):
        if i < 3:
            labels[i] = float(labe[1:])
            # labels[i] = (float(labe[1:]) + 2) * (1 / 22)
        else:
            labels[i] = (float(labe[1:]))
    if n_out == 2:
        return torch.FloatTensor([labels[0]] + [labels[2]])
    elif n_out == 3:

        return torch.FloatTensor([labels[0]] + labels[2:4])


def get_real_label(name, n_out):
    # print(name)
    labels = name.replace('.jpeg', '').split('/')[-1].split('_')
    for i, labe in enumerate(labels[:-1]):
        if i < 3:
            labels[i] = float(labe[1:])
            # labels[i] = (float(labe[1:]) + 2) * (1 / 22)
        else:
            labels[i] = (float(labe[1:]))
    if n_out == 2:
        return torch.FloatTensor([labels[0]] + [labels[2]])
    elif n_out == 3:

        return torch.FloatTensor([labels[0]] + [labels[1]] + [adjust_angle(labels[3])])

