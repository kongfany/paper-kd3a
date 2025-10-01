from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch


def read_domainnet_data(dataset_path, domain_name, split="train"):
    # (Pdb) p dataset_path
    # 'basepath/dataset/DomainNet'
    # (Pdb) p domain_name
    # 'clipart'
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    # (Pdb) p split_file
    # 'basepath/dataset/DomainNet/splits/clipart_train.txt'
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels
# (Pdb) p len(data_paths)
# 33525


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)
    # 这段代码定义了一个名为DomainNet的类，它是一个PyTorch的Dataset类的子类。它包含了以下几个成员函数：
    #
    # __init__：初始化函数，它接收四个参数：data_paths，data_labels，transforms和domain_name。data_paths是一个列表，其中包含所有图片的路径，data_labels是一个列表，其中包含所有图片的标签。transforms是一个包含一系列图像变换的对象，用于对图像进行预处理。domain_name是一个字符串，代表数据集所属的域。
    # __getitem__：数据集的索引函数，它接收一个整数index，并返回该索引位置的图像及其标签。具体操作包括打开指定路径的图像，将其转换为RGB模式，使用transforms对其进行预处理，最后返回预处理后的图像和其标签。
    # __len__：返回数据集的长度，即包含的图像数量。
    # 该类的作用是为数据集提供数据和标签，并在调用训练或测试函数时，返回一个元组包含数据和标签，用于后续的训练或测试。


def get_domainnet_dloader(base_path, domain_name, batch_size, num_workers):
    # (Pdb) p base_path
    # 'basepath'
    # (Pdb) p domain_name
    # 'clipart'
    # (Pdb) p batch_size
    # 50
    # (Pdb) p num_workers
    # 8
    dataset_path = path.join(base_path, 'dataset', 'DomainNet')
    # (Pdb) p dataset_path
    # 'basepath/dataset/DomainNet'
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")

    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    # (Pdb) p len(train_data_paths)
    # 33525
    # (Pdb) p len(test_data_paths)
    # 14604
    # (Pdb) p train_data_paths[0]
    # 'basepath/dataset/DomainNet/clipart/aircraft_carrier/clipart_001_000018.jpg'
    # (Pdb) p train_data_labels[0]
    # 0
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    # 这段代码是用来定义训练集数据增强的操作，具体解释如下：
    #
    # transforms.RandomResizedCrop(224, scale=(0.75, 1))：随机裁剪224x224大小的图片，裁剪比例在0.75到1之间随机选取。
    # transforms.RandomHorizontalFlip()：随机对图片进行水平翻转。
    # transforms.ToTensor()：将图片数据转换成Tensor类型。
    # 这些操作可以增加训练集的多样性，提高模型的泛化能力。
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    # transforms_train = transforms.Compose([
    #     transforms.RandomResizedCrop(96, scale=(0.75, 1)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.Resize((96,96)),
    #     transforms.ToTensor()
    # ])
    # transforms_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor()
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    # (Pdb) p train_dataset
    # <datasets.DomainNet.DomainNet object at 0x7f4128da1250>
    # (Pdb) p len(train_dataset)
    # 33525
    # (Pdb) p train_dataset[0]
    # (tensor([[[0.9608, 0.9843, 0.9882,  ..., 0.9922, 0.9922, 0.9922],
    #          [0.9098, 0.9373, 0.9373,  ..., 0.9608, 0.9412, 0.9333],
    #          [0.9176, 0.9216, 0.9255,  ..., 0.9176, 0.9333, 0.9098],
    #          ...,
    #          [0.9373, 0.9294, 0.9216,  ..., 0.9412, 0.9255, 0.9216],
    #          [0.9255, 0.9255, 0.9333,  ..., 0.9333, 0.9333, 0.9137],
    #          [0.9882, 0.9843, 0.9882,  ..., 0.9804, 0.9882, 0.9804]],
    #
    #         [[0.9608, 0.9843, 0.9882,  ..., 0.9922, 0.9922, 0.9922],
    #          [0.9098, 0.9373, 0.9373,  ..., 0.9608, 0.9412, 0.9333],
    #          [0.9176, 0.9216, 0.9255,  ..., 0.9176, 0.9333, 0.9098],
    #          ...,
    #          [0.9373, 0.9294, 0.9216,  ..., 0.9412, 0.9255, 0.9216],
    #          [0.9255, 0.9255, 0.9333,  ..., 0.9333, 0.9333, 0.9137],
    #          [0.9882, 0.9843, 0.9882,  ..., 0.9804, 0.9882, 0.9804]],
    #
    #         [[0.9608, 0.9843, 0.9882,  ..., 0.9922, 0.9922, 0.9922],
    #          [0.9098, 0.9373, 0.9373,  ..., 0.9608, 0.9412, 0.9333],
    #          [0.9176, 0.9216, 0.9255,  ..., 0.9176, 0.9333, 0.9098],
    #          ...,
    #          [0.9373, 0.9294, 0.9216,  ..., 0.9412, 0.9255, 0.9216],
    #          [0.9255, 0.9255, 0.9333,  ..., 0.9333, 0.9333, 0.9137],
    #          [0.9882, 0.9843, 0.9882,  ..., 0.9804, 0.9882, 0.9804]]]), 0)
    # (Pdb) p type(train_dataset[0])
    # <class 'tuple'>
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    # 这段代码定义了一个DataLoader对象，用于从train_dataset中获取数据并以batch_size大小的批量加载。
    # num_workers表示在数据加载过程中使用的进程数，pin_memory表示是否将数据存储在固定的内存中以提高性能，shuffle表示是否对数据进行随机重排。
    # train_dloader是一个迭代器，可以用于循环遍历整个数据集，每次返回一个batch的数据。
    # (Pdb) p type(train_dloader)
    # <class 'torch.utils.data.dataloader.DataLoader'>
    # 通常建议将num_workers设置为你计算机CPU核心数的一半到全部，
    # 例如如果你的CPU有8个核心，那么可以将num_workers设置为4到8之间的值。但是，这也需要考虑其他因素，例如内存大小和数据集大小等。

    # 得到的 train_dloader 是一个可以迭代的对象，可以通过 for 循环遍历数据集中的每个 batch。在每次迭代中，会返回一个 batch 的数据和标签，这些数据可以传入模型进行训练。
    #
    # 例如，可以使用以下代码对 train_dloader 进行遍历：
    #
    # for i, (images, labels) in enumerate(train_dloader):
    #     # 训练模型
    # 其中 i 表示当前 batch 的索引，images 是一个大小为 [batch_size, channels, height, width] 的张量，表示当前 batch 的图像数据，
    # labels 是一个大小为 [batch_size] 的张量，表示当前 batch 的标签。你可以根据具体需要进行模型训练和优化。
    # (Pdb) p batch_size
    # 50
    # (Pdb) p len(train_dloader)
    # 671
    # (Pdb) p 671*50
    # 33550

    # enumerate 是 Python 中的一个内置函数，用于将一个可迭代对象（如列表、元组、字符串等）组合为一个索引序列，同时列出数据和数据下标。
    # 这里的 train_loader 是一个 PyTorch 中的 DataLoader 对象，它可以把数据集按照指定的 batch_size 进行划分，
    # 每次返回一个包含 batch_size 个数据和标签的元组 (images, labels)，
    # enumerate 函数会自动返回一个从0开始递增的整数下标 i 和对应的 (images, labels) 元组。
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True)
    return train_dloader, test_dloader
