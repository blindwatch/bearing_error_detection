import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from config import cfg
plt.rcParams['font.sans-serif'] = ['SimHei']  # 画图
plt.rcParams['axes.unicode_minus'] = False
category = ['B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007@6', 'OR014@6', 'OR021@6', 'NORMAL']

plt_acc = pd.read_csv(cfg.ACC_PLOT, index_col=0)
acc_train = plt_acc.loc['train'].values
acc_val = plt_acc.loc['val'].values

plt.plot(acc_train, label='训练集')
plt.plot(acc_val, label='验证集')
plt.legend()
plt.xticks(range(0, len(acc_val), 2))
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.title("训练准确率")
plt.savefig(cfg.RESULT_PIC + 'accuracy.png', dpi = 300)
plt.show()

plt_loss = pd.read_csv(cfg.LOSS_PLOT, index_col=0)
loss_train = plt_loss.loc['train'].values
loss_val = plt_loss.loc['val'].values

plt.plot(loss_train, label='训练集')
plt.plot(loss_val, label='验证集')
plt.legend()
plt.xticks(range(0, len(loss_train), 2))
plt.xlabel("迭代次数")
plt.ylabel("损失")
plt.title("训练损失")
plt.savefig(cfg.RESULT_PIC + 'loss.png', dpi = 300)
plt.show()

plot_first = pd.read_csv(cfg.FIRST_EPOCH, index_col=0)
loss_first = plot_first.loc['loss'].values
acc_first = plot_first.loc['acc'].values

plt.plot(loss_first)
plt.xticks(range(0, len(loss_first), 20))
plt.xlabel("批次")
plt.ylabel("损失")
plt.title("第一次迭代的损失率(每个批次）")
plt.savefig(cfg.RESULT_PIC + 'loss_first.png', dpi = 300)
plt.show()
plt.plot(acc_first)
plt.xticks(range(0, len(acc_first), 20))
plt.xlabel("批次")
plt.ylabel("准确率")
plt.title("第一次迭代的准确率(每个批次）")
plt.savefig(cfg.RESULT_PIC + 'acc_first.png', dpi = 300)
plt.show()

plot_cate_train = pd.read_csv(cfg.CATEGORY_ACC_TRAIN, index_col=0)
for i in range(len(category)):
    plt.plot(plot_cate_train.loc[category[i]].values, label=category[i])
plt.legend()
plt.xticks(range(0, len(plot_cate_train.iloc[0]), 2))
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.title("每类分类的准确率(训练集)")
plt.savefig(cfg.RESULT_PIC + "category_acc_train.png", dpi=300)
plt.show()

plot_cate_val = pd.read_csv(cfg.CATEGORY_ACC_VAL, index_col=0)
for i in range(len(category)):
    plt.plot(plot_cate_val.loc[category[i]].values, label=category[i])
plt.legend()
plt.xticks(range(0, len(plot_cate_val.iloc[0]), 2))
plt.xlabel("迭代次数")
plt.ylabel("准确率")
plt.title("每类分类的准确率(测试集)")
plt.savefig(cfg.RESULT_PIC + "category_acc_val.png", dpi=300)
plt.show()
