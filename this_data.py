import torch

def get_datas(path, get_data):
    datas, labels = get_data(path)
    datas = torch.Tensor(datas).type(torch.FloatTensor)
    labels = torch.Tensor(labels).type(torch.LongTensor)
    return datas, labels

def get_loaders(path, batch_size, get_data_tr, 
    get_data_va=None, get_data_tt=None, 
    random=True, ratio=[0.6, 0.2, 0.2], 
):
    da, la = get_datas(path, get_data_tr)
    if get_data_tt is None and get_data_va is None:
        lengths = [int(da.shape[0]*i) for i in ratio]
        tr_da = da[0:lengths[0]]
        va_da = da[lengths[0]:lengths[0]+lengths[1]]
        tt_da = da[lengths[0]+lengths[1]:]
        tr_la = la[0:lengths[0]]
        va_la = la[lengths[0]:lengths[0]+lengths[1]]
        tt_la = la[lengths[0]+lengths[1]:]
    else:
        tr_da = da
        tr_la = la
        if get_data_va is not None:
            va_da, va_la = get_datas(path, get_data_va)
        else:
            va_da = torch.Tensor().type(torch.FloatTensor)
            va_la = torch.Tensor().type(torch.LongTensor)
        if get_data_tt is not None:
            tt_da, tt_la = get_datas(path, get_data_tt)
        else:
            tt_da = torch.Tensor().type(torch.FloatTensor)
            tt_la = torch.Tensor().type(torch.LongTensor)

    tr_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(tr_da, tr_la),
        batch_size=batch_size,
        shuffle=True,
    )
    if va_da.shape[0] < 2:
        va_loader = None
    else:
        va_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(va_da, va_la),
            batch_size=batch_size,
            shuffle=True,
        )
    if tt_da.shape[0] < 2:
        tt_loader = None
    else:
        tt_loader = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(tt_da, tt_la),
            batch_size=batch_size,
            shuffle=True,
        )

    return tr_loader, va_loader, tt_loader
