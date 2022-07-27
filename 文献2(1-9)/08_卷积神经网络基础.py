'''
进行卷积之后C(通道数)，W（宽），H（高）都可能变，当然也可能不变。卷积核通道的数量和输入图片通道的数量是一致的，输出图像的通道数和卷积核的个数是一样的。
'''

import torch

in_channels, out_channels= 5, 10 #输入和输出的通道数
width, height = 100, 100 #图像大小
kernel_size = 3 #卷积核大小
batch_size = 1

#在torch中输入的数据都是小批量的，所以在输入的时候需要指定，一组有多少个张量
#torch.randn（）的作用是标准正态分布中随机取数，返回一个满足输入的batch，通道数，宽，高的大小的张量
input = torch.randn(batch_size,in_channels,width, height)
#torch.nn.Conv2d（输入通道数量，输出通道数量，卷积核大小）创建一个卷积对象
conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)

'''
卷积层还有很多常见的参数：
（1）padding
由上面我们所举的例子可以看出，当一个图片经过3×3的卷积层以后，它的大小会发生改变，长和宽都会缩小一列，那么如果我们想让它经过卷积层以后大小不变怎么办？这个时候就需要用到padding,讲它的作用就是在处理的时候，在原来输出图像外，增加1圈0.这样在进行卷积的时候，输出图像的大小就和输入图像的大小一样了，不信你可以试一下。（当然，如果你愿意也可以增加n圈0）
用PyTorch模拟一下：
'''
import torch

input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]

#将一个列表转换成一个batch5,通道1，长宽5的张量
input = torch.Tensor(input).view(1, 1, 5, 5)
#卷积层padding=1也就是在外面加一圈
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
#定义一个卷积核
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
#我们将自己设置的卷积核权重设置给卷积层的权重
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output.data)

'''
2）stride(步长)
在之前的例子中，我们在进行了一次卷积运算以后，框框需要往右滑动一位，那么这一位就是他的步长，当然你也可以让他直接向右滑动两位，这个时候就需要stride的这个变量了，结果是这样子的。
'''
import torch

input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]

input = torch.Tensor(input).view(1, 1, 5, 5)
#stride=2步长调整为2
conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, bias=False)
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1, 1, 3, 3)
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output.data)

'''
（3）下采样
我们图像的输入可能是非常大的，所以需要的进行的运算非常的多，这个时候我们就需要下采样来对图像进行缩小处理，降低运算的需求。下采样中我们用到最多的就是maxpooling最大池化层。

举个例子，比如我们用一个2×2大小的最大池化层，那么它就会在我们输入的图像上分割成n个2×2大小的方格，并且在每一个方格当中取最大值，就像下图这样。通过这样的一次运算，图像的长和宽都缩小为原来的1/2，当然你也可以用规格更大的最大池化层。当我们做下采样的时候通道数是不变的，长宽会变。
'''
import torch

input = [3,4,6,5,
         2,4,6,8,
         1,6,7,8,
         9,7,4,6, ]
input = torch.Tensor(input).view(1, 1, 4, 4)

#创建一个2x2的最大池化层
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpooling_layer(input)
print(output)
