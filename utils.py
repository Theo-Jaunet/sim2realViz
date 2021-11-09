import argparse
import base64
import io
import math
import numpy
import numpy as np
import torch
from PIL import Image

import torch.nn.functional as F

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
        else:
            labels[i] = (float(labe[1:]))
    if n_out == 2:
        return torch.FloatTensor([labels[0]] + [labels[2]])
    elif n_out == 3:

        return torch.FloatTensor([labels[0]] + labels[2:4])


def adjust_angle(ang):
    if ang < 0:
        return ang + 360
    return ang


def fullDist(coords1, coords2, ang1, ang2):
    ratio = 1 / 22
    pos = F.mse_loss(torch.from_numpy(numpy.array(coords1)) * ratio, torch.from_numpy(numpy.array(coords2)) * ratio)
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
    labels = name.replace('.jpeg', '').split('/')[-1].split('_')
    for i, labe in enumerate(labels[:-1]):
        if i < 3:
            labels[i] = float(labe[1:])
        else:
            labels[i] = (float(labe[1:]))
    if n_out == 2:
        return torch.FloatTensor([labels[0]] + [labels[2]])
    elif n_out == 3:

        return torch.FloatTensor([labels[0]] + labels[2:4])


def get_real_label(name, n_out):
    labels = name.replace('.jpeg', '').split('/')[-1].split('_')
    for i, labe in enumerate(labels[:-1]):
        if i < 3:
            labels[i] = float(labe[1:])
        else:
            labels[i] = (float(labe[1:]))
    if n_out == 2:
        return torch.FloatTensor([labels[0]] + [labels[2]])
    elif n_out == 3:

        return torch.FloatTensor([labels[0]] + [labels[1]] + [adjust_angle(labels[3])])

