# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:48 下午
# @Author: fany
# @File : 处理多维特征的输入.py
# @Software: PyCharm
# @description:
import numpy as np
import torch
import matplotlib.pyplot as plt

xy = np.loadtxt('../diabetes.csv.gz', delimiter=',', dtype=np.float32)
# 取前8列
x_data = torch.from_numpy(xy[:, :-1])
# 取最后1列
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 注意所有输入参数都使用x
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

model = Model()
# torch.nn.Hardsigmoid()    torch.nn.Sigmoid()  torch.nn.ReLU()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1000000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 反馈
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()