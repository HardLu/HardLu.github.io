---
layout: post
title: "论文笔记：Class-Specific Anchor Based and Context-Guided Multi-Class Object Detection in High Resolution Remote Sensing Imagery with a Convolutional Neural Network"
subtitle: 'CACMOD CNN'
author: "WenlSun"
header-img: "img/post-bg.jpg"
header-style: text
tags:
  - Object Detection
  - 笔记
  - pytorch
---

[论文链接](<https://www.mdpi.com/2072-4292/11/3/272>)

# 研究动机(问题)

**提议目标的不适当的锚尺寸**和**描述物体的特征不具有可区分的能力**,是多尺度地物检测中漏检和错检的主要原因.

1. 为解决"多尺度目标检测器中的锚尺寸固定,对于多尺度物体不适应"的问题. 作者在目标检测器中添加了一个特定于类的锚块,根据真实边界框的`IoU`方法,为每个类别学习合适的锚框大小.类特定的锚框可以提供更适合的初始值(`RPN`),以生成覆盖所有类别的尺度和预测边界框,提高召回率.
2. 类特定锚框可以提供比固定锚框更少的上下文信息,尤其是对于小目标而言.这是因为由提议的类特定锚框生成的边界框通常与真实边界框的大小相当,但具有较少的上下文信息,这不利于分类过程. 作者提出将上下文信息整合到原始高级特征中,以增加特征维度并提高分类器的判别能力,从而在目标检测中获得更高的精度.
3. `NMS`可以直接过滤掉包含不同目标的边界框, 从而导致一些未检测到的目标. 文中提出软filter,通过使用`IoU`相关的权重来降低高度重叠的预测框的分数,并在迭代过程中删除具有低分数的框.
4. 使用`Focal Loss`. 训练的损失使用`Focal Loss`来提升难样本的训练.

# 网络结构

![](/img/CACMOD-CNN/fig1.png)



   ## Learn the Class-Specific Anchors Automatically

+ 选择与IOU相关的函数作为计算类特定锚点的距离损失：

$$
\operatorname{loss}=\frac{\sum_{j=0}^{n} d\left(groundtruth_{-}box_{j}, anchor\right)}{n}
$$

其中$n$ 是每个类别的边界框的数量，$d(groundtruth_box, anchor)=1-IOU(groundtruth_box, anchor)$.

+ 集体的算法流程如下：

![](/img/CACMOD-CNN/fig2.png)



![1560334066091](/img/CACMOD-CNN/fig3.png)



## Merge the Context Information to Improve the Feature Representation

![1560334156126](/img/CACMOD-CNN/fig4.png)

`RoI` 池化后的特征通常设置为$7\times 7$.意味着原图中目标需要$112\times 112$ 大小，但是遥感影像中一类目标类别的大小远远小于$112\times 112$ ,尽管`RoI`池化可以通过上采样或下采样来调整特征图的大小，但由于特征表达类别信息的判别能力小，小特征图可能降低目标检测的准确性。文中将上下文信息引入，通过将预测框以中心取其两倍大小的框，然后`ROI`池化后与原始的`ROI`池化后的特征进行拼接，然后进行分类和回归（对于小目标检测很有用）。

## Soft Filter Effective Predicted Boxes to the Classification

主要的创新点石为每个预测边界框的得分添加一个与IoU有关的权重，具体操作如下式：
$$
S_{i}=\left\{\begin{array}{l}{S_{i}, IOU\left(B_{\max }, B_{i}\right)<N_{t}} \\ {S_{i} \times\left(1-I O U\left(B_{\max }, B_{i}\right)\right.}, {\operatorname{IOU}\left(B_{\max }, B_{i}\right) \geq N_{t}}\end{array}\right.
$$
算法流程如下：

![1560335471614](/img/CACMOD-CNN/fig5.png)

## 实验结果

![1560349254391](/img/CACMOD-CNN/fig6.png)

![1560349278007](/img/CACMOD-CNN/fig7.png)

![1560349295674](/img/CACMOD-CNN/fig8.png)

![1560349320299](/img/CACMOD-CNN/fig9.png)