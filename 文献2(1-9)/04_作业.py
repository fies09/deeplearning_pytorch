#coding = utf-8
'''
    标题
    @name:
    @function:
    @author: Mr.Fan
    @date:2021--  
'''
'''
小知识点：可调用对象

如果要使用一个可调用对象，那么在类的声明的时候要定义一个 call()函数就OK了，就像这样

class Foobar:
	def __init__(self):
		pass
	def __call__(self,*args,**kwargs):
		pass

其中参数*args代表把前面n个参数变成n元组，**kwargsd会把参数变成一个词典，举个例子：

 def func(*args,**kwargs):
 	print(args)
 	print(kwargs)

#调用一下
func(1,2,3,4,x=3,y=5)

结果：
(1，2，3，4）
{‘x’:3,‘y’:5}
'''
import  torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):#构造函数
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)#构造对象，并说明输入输出的维数，第三个参数默认为true，表示用到b
    def forward(self, x):
        y_pred = self.linear(x)#可调用对象，计算y=wx+b
        return  y_pred

model = LinearModel()#实例化模型

criterion = torch.nn.MSELoss(size_average=False)
#model.parameters()会扫描module中的所有成员，如果成员中有相应权重，那么都会将结果加到要训练的参数集合上
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)#lr为学习率

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
