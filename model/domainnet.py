import torch.nn as nn
import torch.nn.functional as F
from .resnet import get_resnet
# from model.resnet import get_resnet
import torch

feature_dict = {"resnet18": 512, "resnet34": 512, "resnet50": 2048, "resnet101": 2048}


class DomainNet(nn.Module):
    def __init__(self, backbone, bn_momentum,pretrained=True, data_parallel=True):
        super(DomainNet, self).__init__()
        encoder = get_resnet(backbone,momentumn=bn_momentum,pretrained=pretrained)
        # 在这行代码中，调用了 get_resnet 函数，并传入参数 backbone、momentumn=bn_momentum 和 pretrained=pretrained。这些参数用于配置 get_resnet 函数内部创建的 ResNet 模型。
        #
        # 具体来说，backbone 参数指定了要使用的 ResNet 架构，比如 "resnet18"、"resnet34"、"resnet50" 或 "resnet101"。这将决定模型的层数和参数量。
        #
        # momentumn=bn_momentum 参数用于设置批归一化层（Batch Normalization）的动量值。它控制了批归一化层中统计信息的更新速度。
        #
        # pretrained=pretrained 参数用于指定是否加载预训练的权重。如果设置为 True，则会加载在大规模数据集上预训练的权重，从而初始化模型的参数；如果设置为 False，则会随机初始化模型的参数。
        #
        # 最后，将创建的 encoder 对象赋值给变量 encoder，供后续在 DomainNet 类中使用。
        if data_parallel:
            self.encoder = nn.DataParallel(encoder)
            # 使用数据并行，即在多个GPU上并行地执行模型的前向传播和反向传播。
        else:
            self.encoder = encoder

    def forward(self, x):
        feature = self.encoder(x)
        return feature


class DomainNetClassifier(nn.Module):
    def __init__(self, backbone, classes=126, data_parallel=True):
        super(DomainNetClassifier, self).__init__()
        linear = nn.Sequential()
        # 这段代码创建了一个空的序列容器对象 linear，用于存储神经网络的层。
        # nn.Sequential() 是 PyTorch 中的一个容器类，用于顺序地组合多个层。通过创建一个空的 nn.Sequential 对象，我们可以向其中添加层，并按照添加的顺序依次进行计算。
        # 在这个特定的代码片段中，linear 对象被初始化为空，表示尚未添加任何层。接下来，可以使用 linear.add_module() 方法逐步向容器中添加层。添加的顺序决定了在模型的前向传播过程中，输入数据将如何流经这些层进行变换和计算。
        # 使用 nn.Sequential 可以方便地定义简单的神经网络结构，尤其是在层之间存在线性的顺序关系时。通过按照添加的顺序定义网络层，可以更清晰地表达模型的结构，并方便后续的参数管理和计算操作。
        linear.add_module("fc", nn.Linear(feature_dict[backbone], classes))
        # 这段代码定义了一个线性层（nn.Linear）并将其添加到一个序列容器（nn.Sequential）中。
        # 首先，创建了一个空的序列容器对象 linear，它将用于存储网络的层。接下来，通过调用 linear.add_module() 方法，将一个线性层添加到序列容器中。
        # 具体而言，nn.Linear(feature_dict[backbone], classes) 创建了一个线性层，它将输入特征的维度设置为 feature_dict[backbone]，输出特征的维度设置为 classes这表示线性层将把输入特征映射到输出特征的空间。在这个线性层中，每个输入特征都与一个权重相乘，并加上一个偏置项，然后输出结果。
        # 最后，将创建的线性层添加到序列容器中，使用名称 "fc" 标识该层。这个名称可以用来查找和访问序列容器中的层。
        # 通过使用序列容器，可以将多个层按顺序组合在一起，形成一个神经网络模型。在此示例中，linear 对象包含一个线性层，这样在模型的前向传播过程中，输入数据将通过线性层进行变换。
        if data_parallel:
            self.linear = nn.DataParallel(linear)
        else:
            self.linear = linear

    def forward(self, feature):
        feature = torch.flatten(feature, 1)
        feature = self.linear(feature)
        return feature
# 这段代码定义了一个深度学习模型，包括一个特征提取器（DomainNet）和一个分类器（DomainNetClassifier）。
#
# DomainNet 类是一个特征提取器模型，它基于 ResNet 架构进行特征提取。通过调用 get_resnet 函数获取相应的 ResNet 模型，然后使用 nn.DataParallel 进行数据并行处理（如果 data_parallel=True）。在前向传播过程中，输入数据通过特征提取器得到特征表示。
#
# DomainNetClassifier 类是一个分类器模型，它基于给定的特征维度（由 feature_dict 字典中的 backbone 对应的值确定）和分类类别数量构建一个全连接层（线性层）。同样地，如果 data_parallel=True，则使用 nn.DataParallel 进行数据并行处理。在前向传播过程中，输入特征被展平并通过线性层进行分类预测。
#
# 这两个类共同构成了一个用于域适应（Domain Adaptation）任务的深度学习模型。DomainNet 用于提取输入数据的特征表示，DomainNetClassifier 用于将这些特征表示映射到相应的类别。

if __name__ == '__main__':
    x = torch.randn(50, 3, 224, 224)
    # bcwh
    net = DomainNet("resnet101", 0.1, False, False)
    # print(net)
    y = net(x)
    classi = DomainNetClassifier("resnet101",345,False)
    l = classi(y)
    print(y.shape)
    print(l.shape)
    # torch.Size([50, 2048, 1, 1])
    # torch.Size([50, 2048, 1, 1])
    # torch.Size([50, 345])


# resnet 50
# DomainNet(
#   (encoder): ResNet(
#     (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#     (relu): ReLU(inplace=True)
#     (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#     (layer1): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#           (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer2): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (3): Bottleneck(
#         (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer3): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (3): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (4): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (5): Bottleneck(
#         (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (layer4): Sequential(
#       (0): Bottleneck(
#         (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#         (downsample): Sequential(
#           (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
#           (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         )
#       )
#       (1): Bottleneck(
#         (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#       (2): Bottleneck(
#         (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#         (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
#         (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#         (relu): ReLU(inplace=True)
#       )
#     )
#     (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
#   )
# )
# torch.Size([50, 2048, 1, 1])

