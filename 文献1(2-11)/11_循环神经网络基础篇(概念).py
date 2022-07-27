'''
How to use RNNCell
注意几个参数
输入和隐层（输出）维度
序列长度
批处理大小
注 调用RNNCell这个需要循环，循环长度就是序列长度
'''
import torch

batch_size = 1  # 批处理大小
seq_len = 3     # 序列长度
input_size = 4  # 输入维度
hidden_size = 2 # 隐层维度

cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# (seq, batch, features)
dataset = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(batch_size, hidden_size)

# 这个循环就是处理seq_len长度的数据
for idx, data in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    print('Input size:', data.shape, data)

    hidden = cell(data, hidden)

    print('hidden size:', hidden.shape, hidden)
    print(hidden)

'''
How to use RNN
确定几个参数
input_size和hidden_size: 输入维度和隐层维度
batch_size: 批处理大小
seq_len: 序列长度
num_layers: 隐层数目
注 直接调用RNN这个不用循环
注：如果使用batch_first: if True, the input and output tensors are provided as:(batch_size, seq_len, input_size)
'''
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

# (seqLen, batchSize, inputSize)
inputs = torch.randn(seq_len, batch_size, input_size)
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)

print('Output size:', out.shape)        # (seq_len, batch_size, hidden_size)
print('Output:', out)
print('Hidden size:', hidden.shape)     # (num_layers, batch_size, hidden_size)
print('Hidden:', hidden)


import torch
input_size = 4
hidden_size = 4
batch_size = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 3, 3]    # hello中各个字符的下标
y_data = [3, 1, 2, 3, 2]    # ohlol中各个字符的下标

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data] # (seqLen, inputSize)

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)   # torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
print(inputs.shape, labels.shape)
