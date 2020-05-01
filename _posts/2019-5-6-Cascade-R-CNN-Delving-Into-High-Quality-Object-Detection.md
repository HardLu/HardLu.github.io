---
layout: post
title: "论文笔记：Cascade R-CNN：Delving into High Quality Object Detection"
subtitle: 'Cascade R-CNN'
author: "WenlSun"
header-img: "img/post-bg.jpg"
header-style: text
tags:
  - Object Detection
  - 笔记
  - pytorch
---

[论文链接](https://arxiv.org/abs/1712.00726)，[官方代码(mmdetection中有实现)](https://github.com/zhaoweicai/cascade-rcnn)，[通天塔译文](http://tongtianta.site/paper/27011)，[参考文章1](https://zhuanlan.zhihu.com/p/36095768)，[参考文章2](https://zhuanlan.zhihu.com/p/42553957)，[参考文章3](https://zhuanlan.zhihu.com/p/40207812)

### 出发点

目标检测被构建为：分类+回归 问题进行解决，所以检测问题本身是一个分类问题，但有和分类问题有很大的区别，因为在检测问题中是对所有的候选框进行打分，而在训练过程中是使用IoU阈值来判断正负样本的。因此IoU阈值的选取是一个非常重要的超参数。通常情况使用`IoU=0.5`来决定正负样本，但是这样会带来一些问题。一方面IoU选取的越高，则得到的正样本更接近目标，因此训练出的检测器更加准确，但是一味的提高IoU的阈值会引发两个问题：(1)正样本过少导致训练过程中出现过拟合问题；(2)训练和测试使用不一样的阈值导致评估性能下降。另一方面，IoU阈值选取的越低，得到的正样本更丰富，有利于检测器的训练，但势必导致测试时出现大量的虚检，也即论文中提到的`"close but not correct"`。

![](F:\Projects\wenliangsun.github.io\img\Cascade-R-CNN\img1.png)

作者针对提出的问题进行实验佐证：

图1(c)给出了经过一次回归以后目标候选框和真实目标框匹配的IoU变化，横轴代表回归前，纵轴代表回归后。不同的颜色反映使用不同的IoU阈值训练的检测器。

![](F:\Projects\wenliangsun.github.io\img\Cascade-R-CNN\img2.png)





未完成，待填 。。。。





























