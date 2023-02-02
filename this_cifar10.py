import os
import tarfile
import numpy as np
import pickle

def UnZip(
    data_path = "/root/autodl-pub/cifar-10/cifar-10-python.tar.gz", 
    save_path = "/root/autodl-nas/data/cifar10",
):
    if not os.path.exists(data_path):
        exit()
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    t = tarfile.open(data_path)
    t.extractall(path = save_path)

def unpickle(file):
    with open(file, 'rb') as fo:
        dataset = pickle.load(fo, encoding='bytes')
    datas = dataset[b'data']
    datas = datas.reshape(datas.shape[0], 3, 32, 32)
    labels = dataset[b'labels']
    return datas, labels

def get_data_train(path):
    path_load = os.path.join(path, 'train.npy')
    if not os.path.exists(path_load):
        datas = np.zeros((0, 3, 32, 32))
        labels = []
        for i in range(5):
            file = os.path.join(path, 'data_batch_'+str(i+1))
            data, label = unpickle(file)
            datas = np.concatenate((datas, data), axis=0)
            labels += label
        dict = {'data':datas, 'label':labels}
        np.save(path_load, dict)

    dict = np.load(path_load, allow_pickle=True).item()
    datas = dict['data']
    labels = dict['label']
    return datas, labels

def get_data_valid(path):
    path_load = os.path.join(path, 'valid.npy')
    if not os.path.exists(path_load):
        file = os.path.join(path, 'test_batch')
        datas, labels = unpickle(file)
        dict = {'data':datas, 'label':labels}
        np.save(path_load, dict)

    dict = np.load(path_load, allow_pickle=True).item()
    datas = dict['data']
    labels = dict['label']
    return datas, labels

def get_data_ttest(path):
    path_load = os.path.join(path, 'ttest.npy')
    if not os.path.exists(path_load):
        file = os.path.join(path, 'test_batch')
        datas, labels = unpickle(file)
        dict = {'data':datas, 'label':labels}
        np.save(path_load, dict)

    dict = np.load(path_load, allow_pickle=True).item()
    datas = dict['data']
    labels = dict['label']
    return datas, labels


if __name__ == "__main__":
    UnZip("/root/autodl-pub/cifar-10/cifar-10-python.tar.gz", "/root/autodl-nas/data/cifar10")
