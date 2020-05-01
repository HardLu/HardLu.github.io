---
layout: post
title: 论文笔记：Counting with Focal for Free
subtitle: Counting with Focal for Free
author: WenlSun
header-img: "img/post-paper-bg.jpg"
header-mask: 0.5
tags:
  - 笔记
  - 人群密集估计
---

[论文链接](https://arxiv.org/abs/1903.12206)，[官方代码](https://github.com/shizenglin/Counting-with-Focus-for-Free)

这篇文章和以往人群密度估计的方法使用点注释（`point annotations`）来估计密度图（`density map`）不一样，它重新考虑了点注释的使用，认为点注释比仅仅构建密度图更具有监督的目的。基于此，作者提出了从两个方面来考虑应用点注释作为监督信息。

### 引言


