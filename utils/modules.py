import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        # backbone ouput size: 13*13*512
        # output size: (13-5+2*2)/1 + 1 = 13, channel=512
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        # output size: (13-9+2*4)/1 + 1 = 13, channel=512
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        # output size: (13-13+2*6)/1 + 1 = 13, channel=512
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        # output size: 13*13, channel=512+512+512+512=2048
        x = torch.cat([x, x_1, x_2, x_3], dim=1) 

        

        return x # SPP后面还会用1*1卷积降维到512个channel
