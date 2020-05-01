---
layout: post
title: "MICCAI2019StrucetSeg分割"
subtitle: "总结"
author: Yunfeilu"
header-style: text
tag:
  - 比赛
---
总结
分析数据集：
如果测试数据尺度差异大，那么就需要做很多的缩放操作；分析数据集是很有必要的，首先可以看数据有没有标注问题，其次根据数据的特点设计数据预处理方法，比如说加窗的像素值、需不需要做clahe、resample，数据需不需要resize（精度损失？）；
同时设计数据增强的方法，比如说根据数据的亮度、对比度(如果数据增强中有增加对比度的功能，而且效果还不错的，那是不是就不用对数据做clahe？因为clahe会丢失数据的很多信息)、颜色差异，以及噪声来设计数据增强方法；
同时设计取块方式，针对哪些label多取块，以及如果数据空间中大部分是黑色的背景，数据可不可以crop，这样能够提升训练的效率
直方图匹配？感觉没有必要

数据预处理：
resample，做了实验，发现数据做不做resample，对性能影响不是很大，但是由于实验没有发挥最好的性能，所以不能判定是不是有用，但是由于resample就是对数据进行resize处理，在做数据增强的时候，也会对数据进行resize操作，所以即便用没有resample的数据进行训练，也会有尺度变化，做了resample当然也会有尺度变化；当然resample有一个好处是，如果把数据resample到高分辨的spacing，能够放大数据，让模型学到更精确的信息，但同时，由于测试时，需要resample到一个spacing，再resample回去，会有predict label的精度损失，所以我的结论是，要resample的话，尽量resample到比较高精度的spacing上，且对小label的要看resample不resample的精度差异在多大层面上影响精度。
数据尺度，这个问题很重要，因为就算resample到同一个spacing，但是不同病人的体型、器官大小还是差异较大，所以resample之后的数据尺度还是不一致，而且在学习的时候，数据增强如果有尺度变化的操作，那resample其实在物理层面上就失去了意义
可以看看不做resample的数据训练的时候，做scale和不做scale的测试结果差异，因为做scale会有label精度损失，不知道会不会影响训练结果，当然原始数据也会有标注不准的问题

数据集划分：
很重要，如何划分训练集、验证集、测试集，关乎模型的泛化性能

模型选择：
Deep Supervision 3d Residual Unet
有人用nnUnet，coarse-to-fine的思想，等

超参数选择：可以选择一批具有代表性的小数据集训练，通常能够快速验证所选参数的性能
学习率：非常重要，决定网络学习的效果，以及学习率衰减的时机和程度，决定模型是否收敛，
batch-size：与显存有很大关系，这次的batch-size选了2，那最好就用instance normalization，因为batch-size过小，不能表征整体数据的分布；同时batch-size越大，则需要学习率也要相应的调大
优化器：adam是自适应的优化器，能够快速收敛，sgd收敛较慢，但是能够收敛到更好的解
L2系数，可以控制模型过拟合程度，过大会引起欠拟合，过小可能会引起过拟合
weight decay，与L2系数有着同样的功效，但是原理不同，L2系数是通过每次将权重衰减，达到减小无用权重的效果；tensorflow中有L2正则时，使用adam优化器会导致L2正则效果变差，所以通过额外使用sgd来对L2正则项做梯度下降，adam中不加入正则项  https://zhuanlan.zhihu.com/p/40814046
patch-size：能够尽可能包含多的器官，但是太大会导致batch-size取不了太大，可以考虑对数据进行resize，但会导致label精度损失，这里选择128*128*48，考虑第三维的维度，第三维分辨率较低，所以尽可能选择大一点的size，但是过大的size也不会导致性能继续提升
scale操作：考虑会影响到label精度，特别是小类，所以可以在数据增强过程中做，原始数据和测试数据建议不要做
BN IN GN，batch-size小的情况下，不能用BN，可以用IN，GN效果也比较差
每层节点数，下采样次数，每层卷积块/残差块数目

取块方式
随机取块，在label少的区域多取块，或者直接把小label区域切出来，作为训练数据

损失函数
dice loss + weighted cross entropy，还没有尝试其他的损失（尝试dice + focal loss，小类分不出来，不知道是不是其他原因导致的）

测试
overlap
去除小于一定比例的小区域