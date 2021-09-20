import torch
from torch import nn
from torch.nn import functional as F


def weights_init(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
        nn.init.orthogonal_(layer.weight.data, gain=1)
        layer.bias.data.fill_(0)
    else:
        pass


class Posest2(nn.Module):
    def __init__(self, args):
        super(Posest2, self).__init__()

        self.conv_head = nn.Sequential(

            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.conv_out = None
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        # self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, args.n_out)

        self.relu = nn.ReLU(True)

        # self.tempfc = nn.Linear(256 * 256 * 4, 3)

        self.args = args

    def forward(self, obs):
        # tempfc = nn.Linear(256 * 256 * 4, 3).cuda()
        t = torch.flatten(obs, 1).cuda()
        # # batchsize = t.shape[0]
        # t.requires_grad = True

        # op = tempfc(t)

        x = self.conv_head(obs)
        # print(x.size())
        x = self.avgpool(x)
        # print(x.size())

        x = torch.flatten(x, 1)
        z = x.detach()
        # z = 0
        x = self.relu(self.fc1(x))
        # print(x)

        # x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        # x.requires_grad = True

        # deriv, = torch.autograd.grad(outputs=x, inputs=obs, grad_outputs=torch.ones_like(x),
        #                              # grad_outputs=None,
        #                              # allow_unused=True,
        #                              create_graph=True, retain_graph=True
        #                              )
        #
        # deriv2, = torch.autograd.grad(outputs=deriv, inputs=obs, grad_outputs=torch.ones_like(deriv),
        #                               # grad_outputs=None,
        #                               # allow_unused=True,
        #                               create_graph=True, retain_graph=True
        #                               )

        # print(deriv.mean())

        return x, z  # , deriv2  # X = [x,z,orr]

    def get_conv_out(self):
        return self.conv_out
