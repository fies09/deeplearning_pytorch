# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:32 下午
# @Author: fany
# @File : 梯度下降算法.py
# @Software: PyCharm
# @description:
'''
梯度下降算法
以模型 为例，梯度下降算法就是一种训练参数  到最佳值的一种算法， 每次变化的趋势由 （学习率：一种超参数，由人手动设置调节），以及 的导数来决定，具体公式如下：
注： 此时函数是指所有的损失函数之和
针对模型  的梯度下降算法的公式化简如下：
'''
#  根据化简之后的公式，就可以编写代码，对 w 进行训练，具体代码如下：
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0


def forward(x):
    return x * w


def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
        return cost / len(xs)


def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
        return grad / len(xs)


print('训练前的预测', 4, forward(4))

cost_list = []
epoch_list = []
# 开始训练(100次训练)
for epoch in range(150):
    epoch_list.append(epoch)
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)
    grad_val = gradient(x_data, y_data)
    w -= 0.1 * grad_val
    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)

print('训练之后的预测', 4, forward(4))

# 画图

plt.plot(epoch_list, cost_list)
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()

# 注: Epoch是训练次数，Cost是误差，可以看到随着训练次数的增加，误差越来越小，趋近于0.
