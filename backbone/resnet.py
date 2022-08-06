import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


# 下载权重文件，预训练用
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 构成BasicBlock的3x3和1x1卷积
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # in_planes输入通道数，out_planes输出通道数
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1 # 暂时忽略，用于dilation

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # conv-1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # conv-2，注意到，这里没有设置stride了
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # downsample是增加x通道数用的，因为下面一层的F(x)可能会增加通道数，前面一层无法直接相加
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x # 上一层的信息

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out) # conv-1有ReLU

        out = self.conv2(out)
        out = self.bn2(out) # 注意到这里conv-2没有ReLU

        if self.downsample is not None:
            identity = self.downsample(x) # 没记错的李沐课里说的是用1x1卷积来增加通道数

        out += identity
        out = self.relu(out) # 注意到，联合上一层信息之后在进行ReLU
        return out

# Bottleneck暂时忽略，现在用不上
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out #

# 定义ResNet的整体结构，初步卷积和池化（汇聚）后交给4个层，然后进行池化（汇聚）后用全连接层和softmax输出
# 由于在目标检测中ResNet只是作为特征提取的backbone还没有到输出的时候，于是去掉了最后的池化、FC和softmax
class ResNet(nn.Module):

    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64 # 给输入图像做64次卷积，输出
        # 3=图像的channel，64=输出通道数，假设224输入(224-7+2*3)/2 + 1取下整=112
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 输出大小计算(112 - 3 + 2*1)/2 + 1 = 56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # block有两种，basic和bottle，ResNet-18用的是Basic
        # 64为当前layer的输出通道数
        # layers[0]是一个数值，表示要几个block
        # model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 进行参数的初始化操作
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 用常数初始化
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,（据说能提高模型的性能）
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    # 这里就是搭积木，暂时略过可以
    # self.layer1 = self._make_layer(block, 64, layers[0])
    # _make_layer(block, 128, layers[1], stride=2)
    # _make_layer(block, 256, layers[2], stride=2)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        C_1 = self.conv1(x)
        C_1 = self.bn1(C_1)
        C_1 = self.relu(C_1)
        C_1 = self.maxpool(C_1)

        C_2 = self.layer1(C_1)
        C_3 = self.layer2(C_2)
        C_4 = self.layer3(C_3)
        C_5 = self.layer4(C_4)

        return C_5
            
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        # strict = False as we don't need fc layer params.
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model

def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model

def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__=='__main__':
    #model = torchvision.models.resnet50()
    print("found ", torch.cuda.device_count(), " GPU(s)")
    device = torch.device("cuda")
    model = resnet101(detection=True).to(device)
    print(model)

    input = torch.randn(1, 3, 512, 512).to(device)
    output = model(input)