# -*- coding = utf-8 -*-
# @Time : 2022/5/26 11:17 下午
# @Author: fany
# @File : Logistic实现二分类.py
# @Software: PyCharm
# @description: "逻辑斯蒂回归（分类问题）"
import torch

# 准备数据集
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])

# 设计网络模型
class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)         # 两个参数分别为w和b

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogisticRegressionModel()

# Construct Loss and optimizer
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# Training cycle
for epoch in range(2000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    # print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x_test = torch.Tensor([1.0])
y_test = model(x_test)
print("y_pred = ", y_test.data)
