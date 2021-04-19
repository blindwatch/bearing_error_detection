import scipy.io as scio
import numpy as np
import pandas as pd
import os
np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)

data_tag = ['train', 'val', 'test']
raw_name = ['B007_', 'B014_', 'B021_', 'IR007_', 'IR014_', 'IR021_',
            'OR007@6_', 'OR014@6_', 'OR021@6_', 'NORMAL_']
path = './rawdata/'
n_feature = 13
n_step = 20
window_size = 200
n_sample = int((120000 - window_size) / n_step + 1)
def feature_produce(data):
    #print(data)
    n = data.shape[0]
    T1 = 0
    T2 = 0
    Var = np.std(data, axis=0)
    En = np.linalg.norm(data, ord=2, axis=0) / 10
    K = np.sum(np.square(data), axis=0) / (100 * np.power(En, 4))
    max_vib1 = 0
    max_vib2 = 0
    peak1 = []
    peak2 = []
    vib1 = []
    vib2 = []
    vib_histo1 = np.zeros((1, 5), dtype=np.float32)
    vib_histo2 = np.zeros((1, 5), dtype=np.float32)
    for i in range(1, n - 1):
        if (data[i, 0] > data[i - 1, 0] and data[i, 0] > data[i + 1, 0]) or (
                data[i, 0] < data[i - 1, 0] and data[i, 0] < data[i + 1, 0]):
            T1 = T1 + 1
            peak1.append(data[i, 0])
            if len(peak1) > 1:
                vib = abs(data[i, 0] - peak1[-2])
                vib1.append(vib)
                if vib > max_vib1:
                    max_vib1 = vib

        if (data[i, 1] > data[i - 1, 1] and data[i, 1] > data[i + 1, 1]) or (
                data[i, 1] < data[i - 1, 1] and data[i, 1] < data[i + 1, 1]):
            T2 = T2 + 1
            peak2.append(data[i, 1])
            if len(peak2) > 1:
                vib = abs(data[i, 1] - peak2[-2])
                vib2.append(vib)
                if vib > max_vib2:
                    max_vib2 = vib

    for i in range(len(vib1)):
        ratio = vib1[i] / max_vib1
        if ratio < 0.2:
            vib_histo1[0, 0] = vib_histo1[0, 0] + 1
        elif ratio < 0.4:
            vib_histo1[0, 1] = vib_histo1[0, 1] + 1
        elif ratio < 0.6:
            vib_histo1[0, 2] = vib_histo1[0, 2] + 1
        elif ratio < 0.8:
            vib_histo1[0, 3] = vib_histo1[0, 3] + 1
        else:
            vib_histo1[0, 4] = vib_histo1[0, 4] + 1
    vib_histo1 = vib_histo1 / len(vib1)

    for i in range(len(vib2)):
        ratio = vib2[i] / max_vib2
        if ratio < 0.2:
            vib_histo2[0, 0] = vib_histo2[0, 0] + 1
        elif ratio < 0.4:
            vib_histo2[0, 1] = vib_histo2[0, 1] + 1
        elif ratio < 0.6:
            vib_histo2[0, 2] = vib_histo2[0, 2] + 1
        elif ratio < 0.8:
            vib_histo2[0, 3] = vib_histo2[0, 3] + 1
        else:
            vib_histo2[0, 4] = vib_histo2[0, 4] + 1
    vib_histo2 = vib_histo2 / len(vib2)
    min_vib1 = np.min(np.array(vib1))
    min_vib2 = np.min(np.array(vib2))
    impulse = np.max(np.abs(data), axis=0) / np.mean(np.abs(data), axis=0)
    peak_mean1 = np.mean(np.abs(np.array(peak1)))
    peak_mean2 = np.mean(np.abs(np.array(peak2)))
    #print(T1, T2)
    #print("var={}".format(Var))
    #print(En)
    #print("max={},{}".format(max_vib1,max_vib2))
    #print(vib_histo1, vib_histo2)
    feature = [T1, T2, Var[0], Var[1], En[0], En[1], vib_histo1[0, 0], vib_histo1[0, 1], vib_histo1[0, 2],
               vib_histo1[0, 3], vib_histo1[0, 4], vib_histo2[0, 0], vib_histo2[0, 1], vib_histo2[0, 2],
               vib_histo2[0, 3], vib_histo2[0, 4], max_vib1, max_vib2, min_vib1, min_vib2, impulse[0], impulse[1],
               peak_mean1, peak_mean2, K[0], K[1]]
    return np.array(feature, dtype=np.float32)


def feature_generate(raw_data):
    dat = np.zeros((n_sample, 2 * n_feature), dtype=np.float32)
    for j in range(n_sample):
        dat[j] = feature_produce(raw_data[j * 10: j * 10 + window_size, :])
    return dat

#训练集
train_1 = np.zeros((10, n_sample, n_feature * 2 + 1), dtype=np.float32)
for i in range(10):
    train = scio.loadmat(path + raw_name[i] + '0')
    data_key1 = [x for x in train.keys() if 'DE' in x]
    data_key2 = [x for x in train.keys() if 'FE' in x]
    train_DE = train[data_key1[0]][:120000]
    train_FE = train[data_key2[0]][:120000]
    raw_train = np.concatenate((train_DE, train_FE), axis=1)
    train_data = feature_generate(raw_train)
    train_label = np.full((n_sample, 1), i)
    train_1[i] = np.column_stack((train_data, train_label))
    #print(raw_name[i])
    #print(train_1[i][:10, :], train_1[i].shape)
    print(1)
train_1796 = train_1.reshape(10 * n_sample, 2 * n_feature + 1)
print(train_1796.shape)

train_2 = np.zeros((10, n_sample, 2 * n_feature + 1), dtype=np.float32)
for i in range(10):
    train = scio.loadmat(path + raw_name[i] + '1')
    data_key1 = [x for x in train.keys() if 'DE' in x]
    data_key2 = [x for x in train.keys() if 'FE' in x]
    train_DE = train[data_key1[0]][:120000]
    train_FE = train[data_key2[0]][:120000]
    raw_train = np.concatenate((train_DE, train_FE), axis=1)
    train_data = feature_generate(raw_train)
    train_label = np.full((n_sample, 1), i)
    train_2[i] = np.column_stack((train_data, train_label))
    #print(train_2[i][:10, :], train_2[i].shape)
    print(2)
train_1770 = train_2.reshape(10 * n_sample, 2 * n_feature + 1)
data_train = np.concatenate((train_1796, train_1770), axis=0)
data_train = pd.DataFrame(data_train, columns=["DEperiod","FEperiod","DEvar", "FEvar", "DEenergy", "FEenergy", "DEb20",
                                               "DEb40", "DEb60", "DEb80", "DEb100", "FEb20", "FEb40", "FEb60", "FEb80",
                                               "FEb100", "DEmax", "FEmax", "DEmin", "FEmin", "DEimpulse", "FEimpulse",
                                               "DEpm", "NEpm", "DEkru", "FEKru","Catergory" ], dtype=np.float32)
data_train.to_csv('./prodata/data_train.csv')


#验证集
data_val = np.zeros((10, n_sample, 2 * n_feature + 1), dtype=np.float32)
for i in range(10):
    val = scio.loadmat(path + raw_name[i] + '3')
    data_key1 = [x for x in val.keys() if 'DE' in x]
    data_key2 = [x for x in val.keys() if 'FE' in x]
    val_DE = val[data_key1[0]][:120000]
    val_FE = val[data_key2[0]][:120000]
    raw_val = np.concatenate((val_DE, val_FE), axis=1)
    val_data = feature_generate(raw_val)
    val_label = np.full((n_sample, 1), i)
    data_val[i] = np.column_stack((val_data, val_label))
    print(3)
data_val = data_val.reshape(10 * n_sample, 2 * n_feature + 1)
data_val = pd.DataFrame(data_val, columns=["DEperiod","FEperiod","DEvar", "FEvar", "DEenergy", "FEenergy", "DEb20",
                                               "DEb40", "DEb60", "DEb80", "DEb100", "FEb20", "FEb40", "FEb60", "FEb80",
                                               "FEb100", "DEmax", "FEmax", "DEmin", "FEmin","DEimpulse", "FEimpulse",
                                                "DEpm", "NEpm", "DEkru", "FEKru","Catergory"], dtype=np.float32)
data_val.to_csv('./prodata/data_val.csv')

#测试集
data_test = np.zeros((10, n_sample, 2 * n_feature + 1), dtype=np.float32)
for i in range(10):
    test = scio.loadmat(path + raw_name[i] + '2')
    data_key1 = [x for x in test.keys() if 'DE' in x]
    data_key2 = [x for x in test.keys() if 'FE' in x]
    test_DE = test[data_key1[0]][:120000]
    test_FE = test[data_key2[0]][:120000]
    raw_test = np.concatenate((test_DE, test_FE), axis=1)
    test_data = feature_generate(raw_test)
    test_label = np.full((n_sample, 1), i)
    data_test[i] = np.column_stack((test_data, test_label))
    print(4)
data_test = data_test.reshape(10 * n_sample, 2 * n_feature + 1)
data_test = pd.DataFrame(data_test, columns=["DEperiod","FEperiod","DEvar", "FEvar", "DEenergy", "FEenergy", "DEb20",
                                               "DEb40", "DEb60", "DEb80", "DEb100", "FEb20", "FEb40", "FEb60", "FEb80",
                                               "FEb100", "DEmax", "FEmax", "DEmin", "FEmin", "DEimpulse", "FEimpulse",
                                               "DEpm", "NEpm", "DEkru", "FEKru","Catergory"], dtype=np.float32)
data_test.to_csv('./prodata/data_test.csv')