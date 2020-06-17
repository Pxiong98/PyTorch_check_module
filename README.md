# PyTorch_check_module
 使用PyTorch框架写的检测神经网络框架模块
## VGGNet
 随着AlexNet在2012年ImageNet大赛上大放异彩后，卷积网络进入了飞速发展阶段，VGG采用了更小的卷积核与更深的网络结构。
 这里搭建了[VGG16](https://github.com/Pxiong98/PyTorch_check_module/blob/master/NeutralModule/VGG.py)的module

## GoogLeNet
 在2014年ImageNet大赛上，获得冠军的Inception v1（又名GooLeNet），这里采用多通道拼接，分别使用1x1,3x3,5x5的卷积获取不同的特征图像，最后再拼接到一起，值得一提的是，这里采用了1x1的卷积核实现了降维的思想。
 这里搭建了[GoogLeNet](https://github.com/Pxiong98/PyTorch_check_module/blob/master/NeutralModule/GoogleNet.py)的module

## ResNet
 在前面神经网络出现后，炼金师开始追求更深层的网络以寻求更优越的性能，但随着网络的加深，一方面会产生梯度消失或者梯度爆炸，另一方面越深的网络返回的梯度相关性越差。
 ResNet提供了一种深度残差框架来解决梯度消失问题，一个残差模块称为Bottleneck，以50层的版本来说，架构具有4个卷积组，每个卷积组分别由3,4,6,3个Bottleneck模块，并且F（x）+x要求通道数相等，所以模块间还需要一个Downsample操作，使得通道数变为相同。
 这里搭建了带有Downsample的[Bottleneck](https://github.com/Pxiong98/PyTorch_check_module/blob/master/NeutralModule/Bottleneck.py)结构
 
## DenseNet
搭建了GrowthRate为32的DenseNet的一个[Block](https://github.com/Pxiong98/PyTorch_check_module/blob/master/NeutralModule/Densenet_block.py)

## FPN
搭建了[FPN](https://github.com/Pxiong98/PyTorch_check_module/blob/master/NeutralModule/FPN.py)网络
