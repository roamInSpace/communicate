import torch
import os
import numpy as np
import torch.nn as nn
import copy

def predict(output):
    pred = torch.max(output, 1)[1]
    return pred

def get_eva(model, loader, device, loss_func):
    cos = 0
    acc = 0
    acc_all = 0
    for (x_i, y_i) in loader:
        x_i = x_i.to(device)
        y_i = y_i.to(device)
        output = model(x_i)
        loss = loss_func(output, y_i)
        cos += loss.item()
        
        y_pred = predict(output).cpu().numpy()
        y_true = y_i.cpu().numpy()
        acc += (y_pred == y_true).astype(int).sum()
        acc_all += y_true.size
    cos = cos/len(loader)
    acc = acc/acc_all
    return cos, acc

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
