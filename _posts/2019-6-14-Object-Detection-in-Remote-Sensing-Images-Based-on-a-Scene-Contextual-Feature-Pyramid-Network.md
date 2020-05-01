---
layout: post
title: "论文笔记：Object Detection in Remote Sensing Images Based on a Scene-Contextual Feature Pyramid Network"
subtitle: 'SCFPN'
author: "WenlSun"
header-img: "img/post-bg.jpg"
header-style: text
tags:
  - Object Detection
  - 笔记
  - pytorch
---

[论文链接](https://www.mdpi.com/2072-4292/11/3/339)

# 研究动机（问题）

**考虑到遥感影像中的目标类型通常与它们所处的场景密切相关**。文中作者通过组合用于目标检测的场景-上下文信息，提出了一种用于遥感影像目标检测的CNN网络。作者提出了三个创新点。

+ 场景-上下文特征金字塔网络。旨在加强目标与场景之间的关系，解决目标变换大引起的问题。
+ 提出一个新的backbone，利用聚合残差模块(`aggregated residual block`)来增加感受野，为目标（尤其是小目标）提供更丰富的信息，提高网络的特征提取能力。
+ 为了进一步提升网络的性能，作者使用组归一化（`Group Normalization`）代替批量归一化`Batch Normalization`，解决了批量归一化的局限性。

与自然图像相比，基于`CNN`的方法具有若干局限性：

+ 在遥感影像中，需要从多个场景（机场，国家，河流等）中检测目标，这增加了目标检测的难度。
+ 与自然图像相比，遥感影像中标注的样本数量较少，这使得网络收敛变得困难。
+ 遥感影像呈透视图，其目标尺度的变化范围要比自然图像大。

统计分析了遥感影像中的目标与场景的相关性：

![1560496622466](/img/SCFPN/fig1.png)

# 提出的方法（Proposal Method）

**`SCFPN` 框架**：`SCFPN`框架由两部分构成：一个基于特征金字塔网络用来生成多尺度的`RoI` 的`RPN`（`FPN-RPN`）和一个用于对`RoI` 分类的场景-上下文特征融合网络。具体来说，在`FPN-RPN`,首先生成每个输入图像的多尺度特征融合的特征图，然后使用`FPN-RPN`生成多尺度的`RoIs`。在场景-上下文特征融合网络中，首先利用骨干网络提取场景上下文和生成多尺度的`RoI`，然后通过组合它们来融合特征，最后使用分类器处理`RoI`的类预测。

![1560496751030](/img/SCFPN/fig2.png)



## SCFPN Framework

![1560498136778](/img/SCFPN/fig3.png)

首先使用`FPN-RPN`网络提取多尺度的`RoI`，然后使用骨干网络提取全局图像特征和多尺度`RoI`特征。为了解决特征尺寸不匹配的问题，使用`RoI Align` 池化来 resize 特征，最后将全局特征和多尺度`RoI`特征进行融合。

![1560500419755](/img/SCFPN/fig4.png)

损失函数：
$$
L\left(\left\{p_{j}\right\},\left\{t_{j}\right\}\right)=\frac{1}{N_{c l s}} \sum_{j} L_{c l s}\left(p_{j}, p_{j}^{*}\right)+\lambda \frac{1}{N_{r e g}} \sum_{j} p_{j}^{*} L_{r e g}\left(t_{j}, t_{j}^{*}\right)
$$

$$
L_{cls}(p, 1)=-\log {pl}
$$

$$
L_{r e g}\left(t_{j}, t_{j}^{*}\right)=\operatorname{smoothL1}\left(t_{j}-t_{j}^{*}\right)
$$

$$
\operatorname{smoothL1}(x)=\left\{\begin{array}{c}{0.5 x^{2}, \text { if }|x|<1} \\ {|x|-0.5, \text { otherwise }}\end{array}\right.
$$



## Backbone Network

作者使用ResNext 块并引入扩张卷积来获得称为ResNext-d的组合结构，其扩大了感受野并增强了对小目标的感知。

![1560500937276](/img/SCFPN/fig5.png)

![1560501015664](/img/SCFPN/fig6.png)

![1560501083254](/img/SCFPN/fig7.png)



## Group Normalization

![1560501137304](/img/SCFPN/fig8.png)



# 实验结果

![1560501182787](/img/SCFPN/fig9.png)

![1560501227044](/img/SCFPN/fig10.png)

![1560501264234](/img/SCFPN/fig11.png)

![1560501304840](/img/SCFPN/fig12.png)

