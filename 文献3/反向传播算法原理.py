# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:35 下午
# @Author: fany
# @File : 反向传播算法原理.py
# @Software: PyCharm
# @description:
'''
注意：1、torch.Tensor()作用是生成新的张量
    2、w.requires_grad = True  意思是：是否需要计算梯度？——True
    3、.item()的作用主要是把数据从tensor取出来，变成python的数据类型
    4、.backward()函数：反向传播求梯度
    5、w一定要取data值进行数值计算，tensor做加法运算会构建运算图，消耗内存
    6、w.grad.data.zero_()  ： 每次反向传播的数据要清零，否则梯度值是每次计算相加的总额

'''
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# torch.Tensor()生成新的张量
w = torch.Tensor([1.0])
# 是否需要计算梯度？——True
w.requires_grad = True


def forward(x):
    return x * w


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


# .item()的作用主要是把数据从tensor取出来，变成python的数据类型
print("训练前的预测是", 4, forward(4).item())
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        # 利用PyTorch的反向传播函数求梯度
        l.backward()
        # 这里是数值计算，w一定要取data值，tensor做加法运算会构建运算图，消耗内存
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - 0.01 * w.grad.data

        # 每次反向传播的数据要清零
        w.grad.data.zero_()
    print("progress:", epoch, l.item())
print("训练之后的预测值是", 4, forward(4).item())