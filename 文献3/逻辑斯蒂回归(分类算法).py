# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:46 下午
# @Author: fany
# @File : 逻辑斯蒂回归(分类算法).py
# @Software: PyCharm
# @description:
import matplotlib.pyplot as plt
import torch
import matplotlib.pyplot as pl
import numpy as np

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    # 构造函数：初始化对象默认调用的函数
    def __init__(self):
        # 必写
        super(LogisticRegressionModel, self).__init__()
        # 构造对象
        self.linear = torch.nn.Linear(1, 1)

    # 前馈任务所要进行的计算
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


# 实例化
model = LogisticRegressionModel()

# 损失函数对象
criterion = torch.nn.BCELoss(reduction='sum')
# 优化器对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 进行训练
for epoch in range(1000):
    # 计算y hat
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 所有权重每次都梯度清零
    optimizer.zero_grad()
    # 反向传播求梯度
    loss.backward()
    # 更新，step（）更新函数
    optimizer.step()

# 画图
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.show()