# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:33 下午
# @Author: fany
# @File : 随机梯度下降算法.py
# @Software: PyCharm
# @description:
'''
随机梯度下降算法
随机梯度下降算法与梯度下降算法的不同之处在于，随机梯度下降算法不再计算损失函数之和的导数，而是随机选取任一随机函数计算导数，随机的决定 w 下次的变化趋势，具体公式变化如图：
'''
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


def gradient(x, y):
    return 2 * x * (x * w - y)


print('训练前的预测', 4, forward(4))

epoch_list = []
loss_list = []
# 开始训练(100次训练)
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w -= 0.01 * grad
        l = loss(x, y)
        loss_list.append(l)
        epoch_list.append(epoch)
        print('Epoch:', epoch, 'w=', w, 'loss=', l)

print('训练之后的预测', 4, forward(4))

# 画图
plt.plot(epoch_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(1)
plt.show()