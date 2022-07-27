#coding = utf-8
'''
    标题
    @name:
    @function:
    @author: Mr.Fan
    @date:2021--  
'''
import matplotlib.pyplot as plt
import numpy as np
# prepare the training set
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# initial guess of weight
w = 1.0 #初始化权重设置为1


# define the model linear model y = w*x
def forward(x):
    return x * w


# define the cost function MSE
def cost(xs, ys):   #计算MSE
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)


# define the gradient function  gd
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

print('Predict (befortraining)',4,forward(4))

mse_list = []
for epoch in range(100):    #计算100次
    # 绘图
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val  # 0.01 learning rate
    mse_list.append(cost_val)
    print('epoch:', epoch, 'w=', w, 'loss=', cost_val)
print('predict (after training)', 4, forward(4))

w_list = np.arange(0,100,1)
plt.plot(w_list, mse_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()