import  torch
import  torch.nn.functional as F
import  numpy as np
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)
    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))#这里需要把原来的输出y传给sigmoid，即实现的区间的映射
        return  y_pred

model = LinearModel()

criterion = torch.nn.BCELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0,10,200)
x_t = torch.Tensor(x).view(200,1)#将数据变成一个二百行一列的矩阵
y_t = model(x_t)
y = y_t.data.numpy()

plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.ylabel('probablility of pass')
plt.xlabel('hours')
plt.grid()#画出网格
plt.show()
