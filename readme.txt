1.config.py 为设置文件，可以设置模型训练参数等（注意重新训练会覆盖权重）

2.data_process.py 为原始数据处理过程（rawdata -> prodata）

3.main.py 为训练或者测试的函数，定义了模型和训练,和数据的写入

4.result.py 用来输出图像结果，存在result文件夹中

4.cache 存储了模型的权重

5.DATASCIENCE 是我最初用来观察波形的图像

6.acc_plot.csv 以迭代次数为横轴，显示准确率对比，loss_plot同理

7.category_acc_xxx 以迭代次数为横轴，显示各类的准确率对比

8.first_epoch 以batch为轴，展示了第一次迭代中的正确率和损失变化
