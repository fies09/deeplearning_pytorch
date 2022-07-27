# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:30 下午
# @Author: fany
# @File : p2作业2.py
# @Software: PyCharm
# @description: "作业内容：尝试使用 y = x *w + b 模型 (两个未知数 w、b），画出3D曲面图"
import numpy as np
import matplotlib.pyplot as plt

# 准确模型是y=2x+1
x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [3.0, 5.0, 7.0, 9.0]


def forward(x):
    return x * W + B


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w = np.arange(0.0, 4.1, 0.1)
b = np.arange(0.0, 2.1, 0.1)
# 将w，b变成二位矩阵
[W, B] = np.meshgrid(w, b)

l_sum = 0
for x_val, y_val in zip(x_data, y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val, y_val)
    l_sum += loss_val

# 引入matplotlib 3D画图
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# surface中的数必须是二维矩阵
ax.plot_surface(W, B, l_sum / 3)
plt.show()