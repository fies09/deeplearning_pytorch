# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:37 下午
# @Author: fany
# @File : 反向传播算法练习.py
# @Software: PyCharm
# @description:
'''
如图所示函数模型，数据集：x_list[] = [1, 2, 3]，y_list[] = [2, 4, 6]
（1）画出计算图；
（2）利用PyTorch实现代码。
'''
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# torch.Tensor()生成新的张量
w1 = torch.Tensor([1.0])
w1.requires_grad = True

w2 = torch.Tensor([1.0])
w2.requires_grad = True

b = torch.Tensor([1.0])
# 是否需要计算梯度？——True
b.requires_grad = True


def forward(x):
    return w1 * x ** 2 + w2 * x + b


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
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        # 这里是数值计算，w一定要取data值，tensor做加法运算会构建运算图，消耗内存
        w1.data = w1.data - 0.01 * w1.grad.data
        w1.grad.data.zero_()

        w2.data = w2.data - 0.01 * w2.grad.data
        w2.grad.data.zero_()

        b.data = b.data - 0.01 * b.grad.data
        # 每次反向传播的数据要清零
        b.grad.data.zero_()
    print("progress:", epoch, l.item())
print("训练之后的预测值是", 4, forward(4).item())