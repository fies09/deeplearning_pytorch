#coding = utf-8
'''
    标题
    @name:
    @function:
    @author: Mr.Fan
    @date:2021--  
'''
'''
1、逻辑斯蒂回归和线性模型的明显区别是在线性模型的后面，添加了激活函数(非线性变换)
2、分布的差异：KL散度，cross-entropy交叉熵
预测与标签越接近，BCE损失越小。

代码说明：

1、视频中代码F.sigmoid(self.linear(x))会引发warning，此处更改为torch.sigmoid(self.linear(x))

     torch.sigmoid() 与 torch.nn.Sigmoid() 对比  

     torch.sigmoid()、torch.nn.Sigmoid()和torch.nn.functional.sigmoid()三者之间的区别

2、BCELoss - Binary CrossEntropyLoss 

     BCELoss 是CrossEntropyLoss的一个特例，只用于二分类问题，而CrossEntropyLoss可以用于二分类，也可以用于多分类。

     如果是二分类问题，建议BCELoss

   五分钟理解：BCELoss 和 BCEWithLogitsLoss的区别

    pytorch nn.BCELoss()详解

   torch.empty()和torch.Tensor.random_()的使用举例   
'''
# import torch
#
# # import torch.nn.functional as F
#
# # prepare dataset
# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[0], [0], [1]])
#
#
# # design model using class
# class LogisticRegressionModel(torch.nn.Module):
#     def __init__(self):
#         super(LogisticRegressionModel, self).__init__()
#         self.linear = torch.nn.Linear(1, 1)
#
#     def forward(self, x):
#         # y_pred = F.sigmoid(self.linear(x))
#         y_pred = torch.sigmoid(self.linear(x))
#         return y_pred
#
#
# model = LogisticRegressionModel()
#
# # construct loss and optimizer
# # 默认情况下，loss会基于element平均，如果size_average=False的话，loss会被累加。
# criterion = torch.nn.BCELoss(size_average=False)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# # training cycle forward, backward, update
# for epoch in range(1000):
#     y_pred = model(x_data)
#     loss = criterion(y_pred, y_data)
#     print(epoch, loss.item())
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
# print('w = ', model.linear.weight.item())
# print('b = ', model.linear.bias.item())
#
# x_test = torch.Tensor([[4.0]])
# y_test = model(x_test)
# print('y_pred = ', y_test.data)

'''
关于BCE loss写了几行代码，帮助理解。 
target 中的数据需要是浮点型
'''
import math
import torch

pred = torch.tensor([[-0.2], [0.2], [0.8]])
target = torch.tensor([[0.0], [0.0], [1.0]])

sigmoid = torch.nn.Sigmoid()
pred_s = sigmoid(pred)
print(pred_s)
"""
pred_s 输出tensor([[0.4502],[0.5498],[0.6900]])
0*math.log(0.4502)+1*math.log(1-0.4502)
0*math.log(0.5498)+1*math.log(1-0.5498)
1*math.log(0.6900) + 0*log(1-0.6900)
"""
result = 0
i = 0
for label in target:
    if label.item() == 0:
        result += math.log(1 - pred_s[i].item())
    else:
        result += math.log(pred_s[i].item())
    i += 1
result /= 3
print("bce：", -result)
loss = torch.nn.BCELoss()
print('BCELoss:', loss(pred_s, target).item())