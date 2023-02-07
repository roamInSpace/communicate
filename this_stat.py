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

def save_classify(content, sheet_number, path, name):
    import xlwt
    import time
    workbook = xlwt.Workbook(encoding = 'utf-8')
    worksheet = workbook.add_sheet(str(sheet_number))

    for i_h, h_i in enumerate(content):
        for i_w, w_i in enumerate(h_i):
            worksheet.write(i_h, i_w, label=w_i)
    
    for _ in range(60):
        try:
            workbook.save(os.path.join(path, name))
            break
        except:
            time.sleep(1)        

def save_info(path, model_best, train_evas, valid_evas, ttest_details):
    torch.save(model_best, os.path.join(path, 'model_best.pkl'))
    save_classify(train_evas+valid_evas, 0, path, 'train&valid.xls')
    save_classify(ttest_details, 0, path, 'ttest.xls')
    pass
