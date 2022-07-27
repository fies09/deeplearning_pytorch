# -*- coding = utf-8 -*-
# @Time : 2022/5/26 10:38 下午
# @Author: fany
# @File : pytorch实现反向传播.py
# @Software: PyCharm
# @description:
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


class LinearModel(torch.nn.Module):
    # 构造函数：初始化对象默认调用的函数
    def __init__(self):
        # 必写
        super(LinearModel, self).__init__()
        # 构造对象
        self.linear = torch.nn.Linear(1, 1)

    # 前馈任务所要进行的计算
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

def closure():
    optimizer.zero_grad()
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    loss.backward()
    return loss

# 实例化
model = LinearModel()

# 损失函数对象
criterion = torch.nn.MSELoss(size_average=False)
# 优化器对象
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 进行训练
for epoch in range(100):
    # 计算y hat
    y_pred = model(x_data)
    # 计算损失
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # 所有权重每次都梯度清零
    optimizer.zero_grad()
    # 反向传播求梯度
    loss.backward()
    # # 更新，step（）更新函数
    # optimizer.step()
    # 传入闭包closure
    optimizer.step(closure)

# 输出权重和偏置
print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

# 测试
x_test = torch.Tensor([[4.0]])
y_test = torch = model(x_test)
print('y_pred=', y_test.data)