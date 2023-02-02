import os
from this_cifar10 import get_data_train, get_data_valid, get_data_ttest
from this_data import get_loaders
from this_model import CNN1
path = "/root/autodl-nas/data/cifar10/cifar-10-batches-py"

train_loader, ttest_loader, valid_loader = get_loaders(
    path, 32, get_data_train, 
    get_data_va=get_data_valid, 
    get_data_tt=get_data_ttest, 
)