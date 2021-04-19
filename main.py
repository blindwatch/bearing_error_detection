import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.pyplot as plt
import random
from config import cfg

plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图
plt.rcParams['axes.unicode_minus'] = False
def readfile(path):
    data = pd.read_csv(path)
    data = data.iloc[:, 1:].to_numpy()
    X = data[:, :-1].astype(np.float32)
    Y = data[:, -1].reshape(-1, 1).astype(np.float32)
    return X, Y


train_x, train_y = readfile(cfg.DATASET + 'data_train.csv')
val_x, val_y = readfile(cfg.DATASET + 'data_val.csv')
test_x, test_y = readfile(cfg.DATASET + 'data_test.csv')
train_val_x = np.concatenate((train_x, val_x), axis=0)
train_val_y = np.concatenate((train_y, val_y), axis=0)


class VibDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X


class BPnet(nn.Module):
    def __init__(self, layers):
        super(BPnet, self).__init__()
        self.fc1 = nn.Linear(layers[0], layers[1])
        self.bn1 = nn.BatchNorm1d(layers[1])
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(layers[1], layers[2])
        self.bn2 = nn.BatchNorm1d(layers[2])
        self.fc3 = nn.Linear(layers[2], layers[3])
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.tanh(out)
        # out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.tanh(out)
        # out = self.relu(out)
        out = self.fc3(out)
        # out = self.soft(out)
        return out


train_set = VibDataset(train_x, train_y)
train_val_set = VibDataset(train_val_x, train_val_y)
val_set = VibDataset(val_x, val_y)
test_set = VibDataset(test_x, test_y)

train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
train_val_loader = DataLoader(train_val_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

random.seed(1)
loss = nn.CrossEntropyLoss()

plt_train_loss = []
plt_val_loss = []
plt_train_acc = []
plt_val_acc = []

plt_category_acc_train = [[], [], [], [], [], [], [], [], [], []]
plt_category_acc_val = [[], [], [], [], [], [], [], [], [], []]
first_epoch_acc = []
first_epoch_loss = []


# start = time.time()
# for x_test, y_test in train_loader:
#   print(x_test.shape)
#   print('%.2f sec' % (time.time() - start))

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
# print("Optimizer's state_dict:")
# for var_name in optimizer.state_dict():
#   print(var_name, "\t", optimizer.state_dict()[var_name])

if cfg.STATE == 'Train':
    epoch = 0
    train_loss = 0
    model = BPnet(cfg.NET_SET).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.DEFAULT_LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1, last_epoch=-1)
    if cfg.TRAIN.FROM_CHECK:
        try:
            if not os.path.exists(cfg.WEIGHTS):
                raise EnvironmentError("CANT FIND WEIGHT")
            checkpoint = torch.load(cfg.WEIGHTS)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            train_loss = checkpoint['loss']
            # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except EnvironmentError as e:
            print("TRAINING FROM BEGIN")

    max_val = 0
    while epoch < cfg.TRAIN.NUM_ITERATION:
        if cfg.TRAIN.MIXVAL:
            if epoch > cfg.TRAIN.MIXEPOCH['epoch']:
                t_l = train_val_loader
            else:
                t_l = train_loader
        else:
            t_l = train_loader
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        train_acc_d = np.zeros((1, 10))
        val_acc_d = np.zeros((1, 10))
        # print(train_set.__len__())
        model.train()
        for i, data in enumerate(t_l):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda().squeeze())
            batch_loss.backward()
            optimizer.step()
            # print(batch_loss)
            # print(data[1].squeeze().shape)
            # print(train_pred.shape)
            # print(np.argmax(train_pred.cpu().data.numpy(), axis=1))
            # print(np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze()))
            train_loss += batch_loss.item()
            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze())
            for k in range(train_pred.shape[0]):
                if np.argmax(train_pred[k].cpu().data.numpy()) == data[1].numpy().squeeze()[k]:
                    j = data[1].numpy().squeeze()[k]
                    train_acc_d[0][j] += 1
            if epoch == 0:
                first_epoch_acc.append(np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze()) / train_pred.shape[0])
                first_epoch_loss.append(batch_loss.item())


        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda().squeeze())

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze())
                val_loss += batch_loss.item()

                for k in range(val_pred.shape[0]):
                    if np.argmax(val_pred[k].cpu().data.numpy()) == data[1].numpy().squeeze()[k]:
                        j = data[1].numpy().squeeze()[k]
                        val_acc_d[0][j] += 1

            # print(train_acc, train_set.__len__())
            if cfg.TRAIN.MIXVAL and epoch > cfg.TRAIN.MIXEPOCH['epoch'] :
                plt_train_acc.append(train_acc / train_val_set.__len__())
                plt_train_loss.append(train_loss / train_val_set.__len__() * cfg.TRAIN.BATCH_SIZE)
                plt_val_acc.append(val_acc / val_set.__len__())
                plt_val_loss.append(val_loss / val_set.__len__() * cfg.TRAIN.BATCH_SIZE)
                for k in range(10):
                    plt_category_acc_train[k].append("{:.4f}".format(train_acc_d[0][k] / train_val_set.__len__() * 10))
                    plt_category_acc_val[k].append("{:.4f}".format(val_acc_d[0][k] / val_set.__len__() * 10))
                print('Train_val[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
                      (epoch + 1, cfg.TRAIN.NUM_ITERATION, time.time() - epoch_start_time, plt_train_acc[-1],
                       plt_train_loss[-1],
                       plt_val_acc[-1], plt_val_loss[-1]))
            else:
                plt_train_acc.append(train_acc / train_set.__len__())
                plt_train_loss.append(train_loss / train_set.__len__() * cfg.TRAIN.BATCH_SIZE)
                plt_val_acc.append(val_acc / val_set.__len__())
                plt_val_loss.append(val_loss / val_set.__len__() * cfg.TRAIN.BATCH_SIZE)
                print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' % \
                      (epoch + 1, cfg.TRAIN.NUM_ITERATION, time.time() - epoch_start_time, plt_train_acc[-1],
                       plt_train_loss[-1],
                       plt_val_acc[-1], plt_val_loss[-1]))
                # with open('cjy.txt', "a") as file:  # 只需要将之前的”w"改为“a"即可，代表追加内容
                #    for k in range(10):
                #        file.write("{:.4f}".format(train_acc_d[0][k] / train_set.__len__() * 10) + "  ")
                #    file.write("\n")
                for k in range(10):
                    plt_category_acc_train[k].append("{:.4f}".format(train_acc_d[0][k] / train_set.__len__() * 10))
                    plt_category_acc_val[k].append("{:.4f}".format(val_acc_d[0][k] / val_set.__len__() * 10))

            if  val_acc > max_val:
                max_val = val_acc
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    # 'scheduler_state_dict': scheduler.state_dict()
                }, cfg.WEIGHTS_BEST)

            if (epoch + 1) % cfg.TRAIN.SAVE_FREQ == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': train_loss,
                    # 'scheduler_state_dict': scheduler.state_dict()
                }, cfg.WEIGHTS)
        # scheduler.step()
        epoch = epoch + 1

    first_epoch = pd.DataFrame(data=[first_epoch_loss, first_epoch_acc], index=['loss', 'acc'])
    first_epoch.to_csv(cfg.FIRST_EPOCH)
    loss_plot = pd.DataFrame(data=[plt_train_loss, plt_val_loss], index=['train', 'val'])
    loss_plot.to_csv(cfg.LOSS_PLOT)
    acc_plot = pd.DataFrame(data=[plt_train_acc, plt_val_acc], index=['train', 'val'])
    acc_plot.to_csv(cfg.ACC_PLOT)
    category_acc_train = pd.DataFrame(data=plt_category_acc_train,
                                      index=['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021',
                                             'OR007@6', 'OR014@6', 'OR021@6', 'NORMAL'])
    category_acc_train.to_csv(cfg.CATEGORY_ACC_TRAIN)
    category_acc_val = pd.DataFrame(data=plt_category_acc_val,
                                    index=['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021',
                                                                      'OR007@6', 'OR014@6', 'OR021@6', 'NORMAL'])
    category_acc_val.to_csv(cfg.CATEGORY_ACC_VAL)

else:
    if not os.path.exists(cfg.WEIGHTS_BEST):
        raise EnvironmentError("MODEL NOT TRAINED")
    else:
        test_loss = []
        test_acc = []
        test_acc_d = np.zeros((1,10))
        model = BPnet(cfg.NET_SET).cuda()
        checkpoint = torch.load(cfg.WEIGHTS_BEST)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                test_pred = model(data[0].cuda())
                batch_loss = loss(test_pred, data[1].cuda().squeeze())

                test_acc.append(np.sum(np.argmax(test_pred.cpu().data.numpy(), axis=1) == data[1].numpy().squeeze()))
                test_loss.append(batch_loss.item())

                for k in range(test_pred.shape[0]):
                    if np.argmax(test_pred[k].cpu().data.numpy()) == data[1].numpy().squeeze()[k]:
                        j = data[1].numpy().squeeze()[k]
                        test_acc_d[0][j] += 1
            test_acc_d = test_acc_d / test_set.__len__() * 10

        plt.plot(test_loss)
        plt.title('测试损失')
        plt.savefig(cfg.RESULT_PIC + 'TEST_loss.png')
        plt.show()

        plt.plot(test_acc)
        plt.title('测试准确率')
        plt.savefig(cfg.RESULT_PIC + 'test_acc.png')
        plt.show()

        print(test_acc_d)
        print(np.sum(test_acc_d) / 10)