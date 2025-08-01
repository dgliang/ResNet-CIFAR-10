import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from utils.cutout import Cutout
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import torch.nn.functional as F


# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# number of subprocesses to use for data loading
num_workers = 2
# 每批加载图数量
batch_size = 16
# percentage of training set to use as validation
valid_size = 0.2

def read_dataset(batch_size=batch_size, valid_size=valid_size, num_workers=num_workers, pic_path='dataset'):
    """
    batch_size: Number of loaded drawings per batch
    valid_size: Percentage of training set to use as validation
    num_workers: Number of subprocesses to use for data loading
    pic_path: The path of the pictrues
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(x.unsqueeze(0), (4,4,4,4), mode="reflect").squeeze()),
        transforms.ToPILImage(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], 
                             std=[0.24703233, 0.24348505, 0.26158768]),
        Cutout(n_holes=1, length=5),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124], 
                             std=[0.24703233, 0.24348505, 0.26158768]),
    ])


    # 将数据转换为torch.FloatTensor，并标准化。
    train_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_train)
    valid_data = datasets.CIFAR10(pic_path, train=True,
                                download=True, transform=transform_test)
    test_data = datasets.CIFAR10(pic_path, train=False,
                                download=True, transform=transform_test)
        

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    # random indices
    np.random.shuffle(indices)
    # the ratio of split
    split = int(np.floor(valid_size * num_train))
    # divide data to radin_data and valid_data
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    # 无放回地按照给定的索引列表采样样本元素
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=True)

    return train_loader,valid_loader,test_loader
