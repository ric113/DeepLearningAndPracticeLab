# Lab1

## Objective
Build the state-of-the-art convolutional neural network architecture: Residual Network (ResNet) and train it on the Cifar-10 dataset .
* [Lab1 Deep residual learning](Lab1_deep%20residual%20learning.pdf)

## Report
* [Lab1 Report](./Report.pdf)

## Resources
* [pytorch-cifar Resnet](https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py)
* [给妹纸的深度学习教学(4)——同Residual玩耍](https://zhuanlan.zhihu.com/p/28413039)
* [Deep Residual Networks学习(一)](https://zhuanlan.zhihu.com/p/22071346)
    - 『 虚线部分均处于维度增加部分，亦即卷积核数目倍增的过程，这时进行F(x)+x就会出现二者维度不匹配，这里论文中采用两种方法解决这一问题(其实是三种，但通过实验发现第三种方法会使performance急剧下降，故不采用):
        - A.zero_padding:对恒等层进行0填充的方式将维度补充完整。这种方法不会增加额外的参数
        - B.projection:在恒等层采用1x1的卷积核来增加维度。这种方法会增加额外的参数 』