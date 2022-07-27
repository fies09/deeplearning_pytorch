#coding = utf-8
# -*- coding:utf-8 -*-
'''
    标题
    @name:
    @function:
    @author: Mr.Fan
    @date:2021--  
'''
'''
PyTorch Fashion(风格)

1、prepare dataset

2、design model using Class  # 目的是为了前向传播forward，即计算y hat(预测值)

3、Construct loss and optimizer (using PyTorch API) 其中，计算loss是为了进行反向传播，optimizer是为了更新梯度。

4、Training cycle (forward,backward,update)

代码说明：

1、Module实现了魔法函数__call__()，call()里面有一条语句是要调用forward()。因此新写的类中需要重写forward()覆盖掉父类中的forward()

2、call函数的另一个作用是可以直接在对象后面加()，例如实例化的model对象，和实例化的linear对象

3、本算法的forward体现是通过以下语句实现的：

y_pred = model(x_data)
由于魔法函数call的实现,model(x_data)将会调用model.forward(x_data)函数，model.forward(x_data)函数中的

y_pred = self.linear(x)
self.linear(x)也由于魔法函数call的实现将会调用torch.nn.Linear类中的forward，至此完成封装，也就是说forward最终是在torch.nn.Linear类中实现的，具体怎么实现，可以不用关心，大概就是y= wx + b。

关于魔法函数call在PyTorch中的应用的进一步解释：

pytorch 之 __call__, __init__,forward

pytorch系列nn.Modlue中call的进一步解释

4、本算法的反向传播，计算梯度是通过以下语句实现的：

loss.backward() # 反向传播，计算梯度
5、本算法的参数(w,b)更新，是通过以下语句实现的：

optimizer.step() # update 参数，即更新w和b的值
6、 每一次epoch的训练过程，总结就是

①前向传播，求y hat （输入的预测值）

②根据y_hat和y_label(y_data)计算loss

③反向传播 backward (计算梯度)

④根据梯度，更新参数

7、本实例是批量数据处理，小伙伴们不要被optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)误导了，以为见了SGD就是随机梯度下降。要看传进来的数据是单个的还是批量的。这里的x_data是3个数据，是一个batch，调用的PyTorch API是 torch.optim.SGD，但这里的SGD不是随机梯度下降，而是批量梯度下降。也就是说，梯度下降算法使用的是随机梯度下降，还是批量梯度下降，还是mini-batch梯度下降，用的API都是 torch.optim.SGD。

8、torch.nn.MSELoss也跟torch.nn.Module有关，参与计算图的构建，torch.optim.SGD与torch.nn.Module无关，不参与构建计算图。

9、传送门 torch.nn.Linear的pytorch文档
'''
import torch

# prepare dataset
# x,y是矩阵，3行1列 也就是说总共有3个数据，每个数据只有1个特征
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])


#design model using class
'''
our model class should be inherit from nn.Module, which is base class for all neural network modules.
member methods __init__() and forward() have to be implemented
class nn.linear contain two member Tensors: weight and bias
class nn.Linear has implemented the magic method __call__(),which enable the instance of the class can
be called just like a function.Normally the forward() will be called 

'''
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()

# construct loss and optimizer
# criterion = torch.nn.MSELoss(size_average = False)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # model.parameters()自动完成参数的初始化操作

# training cycle forward, backward, update
for epoch in range(100):
    y_pred = model(x_data)  # forward:predict
    loss = criterion(y_pred, y_data)  # forward: loss
    print(epoch, loss.item())

    optimizer.zero_grad()  # the grad computer by .backward() will be accumulated. so before backward, remember set the grad to zero
    loss.backward()  # backward: autograd，自动计算梯度
    optimizer.step()  # update 参数，即更新w和b的值

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)