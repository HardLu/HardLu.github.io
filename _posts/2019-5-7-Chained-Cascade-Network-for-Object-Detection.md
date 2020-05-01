---
layout: post
title: "论文笔记：Chained Cascade Network for Object Detection"
subtitle: 'Cascade Network'
author: "WenlSun"
header-img: "img/post-bg.jpg"
header-style: text
tags:
  - Object Detection
  - 笔记
  - pytorch
---



作者设计的网络在训练和预测阶段都有层级网络，可以在节省训练和预测时间的情况下提升检测效果。因为在浅层网络拒绝RoI在深层网络不会用来被使用。 前阶段的网络用来学简单的分类，可以去除那些比较容易区分的背景，而深层用来学难的分类。以图为例，第一层，学习是不是哺乳动物。第二层学习是哪种动物如羊牛猪。最后一层学习区分都有角的动物，如牛羊。



作者的创新主要可以归于以下几点：把级联分类器用在物体检测中，分类器级联中在上一层的分类得分在下一层也会被用到。不同层用到不同的特征，这些特征可以是不同深度、不同参数、分辨率和语义信息。上一层的特征可以被用到当前层，作为本层的先验知识。而所有的处理，如框回归、特征级联、分类级联都是在一个完整的网络中可以E2E处理。