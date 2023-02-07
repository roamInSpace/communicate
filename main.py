import os
import torch
from this_cifar10 import get_data_train, get_data_valid, get_data_ttest
from this_data import get_loaders
from this_model import train, CNN1
from this_stat import get_parameter_number, save_info
path = "/root/autodl-nas/data/cifar10/cifar-10-batches-py"

train_loader, ttest_loader, valid_loader = get_loaders(
    path, 32, get_data_train, 
    get_data_va=get_data_valid, 
    get_data_tt=get_data_ttest, 
)

path_save = os.path.join(path, 'CNN')
if not(os.path.exists(path_save)):
    os.makedirs(path_save)
model = CNN1(3, 10)
print(get_parameter_number(model))
torch.cuda.set_device(0)
device = torch.device('cuda')
model.to(device)
model_best, train_evas, valid_evas, ttest_details = train(
    device, model, 0.001, train_loader, 
    valid_loader, [ttest_loader], 
)
save_info(path_save, model_best, train_evas, valid_evas, ttest_details[0])