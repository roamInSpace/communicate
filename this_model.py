import numpy as np
import torch
import torch.nn as nn
import copy
from this_stat import get_eva, predict

class CNN1(nn.Module):
    def __init__(self, channel, cla_num):
        super(CNN1, self).__init__()

        self.cla_num = cla_num

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, 32, 5, 1, 2), 
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2), 
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2), 
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 1, 2), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, 5, 1, 2), 
            nn.BatchNorm2d(512), 
            nn.ReLU(), 
            nn.MaxPool2d(2), 
        )

        self.out = nn.Sequential(
            nn.Linear(256*2*2, 100), 
            nn.ReLU(), 
            nn.Linear(100, cla_num), 
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.conv5(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output



def train(device, model, lr, train_loader, 
    valid_loader=None, ttest_loaders=None, 
    loss_weight=None, stop_num=10, stop_acc=True, 
):
    model_best = None
    train_evas = [[], []]
    valid_evas = [[], []]
    ttest_details = []

    #region 训练
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weight=loss_weight)
    loss_func.to(device)
    eva_best = -float('inf')
    epoch_best = 0
    i_epoch = 0
    while(True):
        i_epoch += 1
        print('epoch'+str(i_epoch))

        model.train()
        for (x_i, y_i) in train_loader:
            x_i = x_i.to(device)
            y_i = y_i.to(device)
            output = model(x_i)
            loss = loss_func(output, y_i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        
        cos, acc = get_eva(model, train_loader, device, loss_func)
        train_evas[0].append(cos)
        train_evas[1].append(acc)
        
        if valid_loader is not None:
            cos, acc = get_eva(model, valid_loader, device, loss_func)
            valid_evas[0].append(cos)
            valid_evas[1].append(acc)
        
        if stop_acc is True:
            eva = acc
        else:
            eva = -cos
        if(eva > eva_best):
            eva_best = eva
            epoch_best = i_epoch
            model_best = copy.deepcopy(model)
        else:
            if i_epoch-epoch_best >= stop_num and i_epoch >= 30:
                break
    #endregion

    #region 测试
    if ttest_loaders is not None:
        for ttest_loader in ttest_loaders:
            ttest_detail = np.zeros((model.cla_num, model.cla_num))
            for (x_i, y_i) in ttest_loader:
                x_i = x_i.to(device)
                y_i = y_i.to(device)
                output = model_best(x_i)
                y_pred = predict(output)
                for (y_i_i, y_pred_i) in zip(y_i, y_pred):
                    ttest_detail[y_i_i, y_pred_i] += 1
            ttest_details.append(ttest_detail)
    #endregion
    
    return model_best, train_evas, valid_evas, ttest_details
