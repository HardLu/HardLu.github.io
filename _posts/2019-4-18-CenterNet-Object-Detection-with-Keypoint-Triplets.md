---
layout: post
title: "论文笔记：CenterNet: Object Detection with Keypoint Triplets"
subtitle: 'Object Detection'
author: "WenlSun"
header-img: "img/post-bg-dreamer.jpg"
header-mask: 0.5
tags:
  - Object Detection
  - 笔记
---

[论文链接](https://arxiv.org/abs/1904.08189)，[官方代码](https://github.com/Duankaiwen/CenterNet)，[参考文章](https://zhuanlan.zhihu.com/p/62789701)

### 引言

传统的基于关键点的目标检测方法例如最具代表性的 CornerNet [^1] 通过检测物体的左上角点和右下角点来确定目标，但在确定目标的过程中，**无法有效利用物体的内部的特征，即无法感知物体内部的信息**，从而导致该类方法产生了很多误检 (错误目标框)。本文利用关键点三元组即中心点、左上角点和右下角点三个关键点而不是两个点来确定一个目标，使网络花费了很小的代价便具备了感知物体内部信息的能力，从而能有效抑制误检。另外，为了更好的检测中心点和角点，我们分别提出了 `center pooling` 和 `cascade corner pooling` 来提取中心点和角点的特征。我们方法的名字叫` CenterNet`，是一种 one-stage 的方法，在最具挑战性之一的数据集 MS COCO上，获得了47% AP，超过了所有已知的 one-stage 检测方 法，并大幅度领先，其领先幅度至少达 4.9%。


### CenterNet 原理

我们抑制误检的原理基于以下推论：如果目标框是准确的，那么在其中心区域能够检测到目标中心点的概率就会很高，反之亦然。因此，首先利用左上和右下两个角点生成初始目标框，对每个预测框定义一个中心区域，然后判断每个目标框的中心区域是否含有中心点，若有则保留该目标框，若无则删除该目标框，其原理如图所示。

![](/img/post-CenterNet-fig1a.png)

### **Baseline 和 Motivation**

其实不光是基于关键点的 one-stage 方法无法感知物体内部信息，几乎所有的 one-stage 方法都存在这一问题。本论文的 baseline 为 CornerNet，因此首先讨论 CornerNet 为什么容易产生很多的误检。首先，CornerNet 通过检测角点确定目标，而不是通过初始候选框 anchor 的回归确定目标，由于没有了 anchor 的限制，使得任意两个角点都可以组成一个目标框，这就对判断两个角点是否属于同一物体的算法要求很高，一但准确度差一点，就会产生很多错误目标框。其次，恰恰这个算法有缺陷。因为此算法在判断两个角点是否属于同一物体时，缺乏全局信息的辅助，因此很容易把原本不是同一物体的两个角点看成是一对角点，因此产生了很多错误目标框。最后，角点的特征对边缘比较敏感，这导致很多角点同样对背景的边缘很敏感，因此在背景处也检测到了错误的角点。综上原因，使得 CornerNet 产生了很多误检。如图所示，我们用 CornerNet 对两张图片进行检测，根据每个预测目标框的 confidence 选出 top100 个预测框 (根据 MS COCO 标准)，可以发现产生了很多误检。其中蓝色框为 ground truth, 红色框为预测框。

![](/img/CenterNet/post-CenterNet-fig1b.png)

为了能够量化的分析误检问题，我们提出了一种新的衡量指标，称为FD (false discovery) rate, 此指标能够很直观的反映出误检情况。FD rate 的计算方式为 FD = 1-AP， 其中 AP 为 IoU 阈值取[0.05 : 0.05 : 0.5]下的平均精度。我们统计了 CornerNet 的误检情况，如表1所示:

![](/img/CenterNet/post-CenterNet-tb1.png)

可以看到，*FD* = 37.8，而 $FD_5$高达32.7，这意味着即使我们把条件限制的很严格：只有那些与 ground-truth 的 $IoU< 0.05$ 的才被认定为错误目标框，每100个预测框中仍然平均有32.7 个错误目标框！而小尺度的目标框其FD更是达到了60.3！

我们分析出了 `CornerNet` 的问题后，接下来就是找出解决之道，关键问题在于让网络具备感知物体内部信息的能力。一个较容易想到的方法是把 `CornerNet` 变成一个 two-stage 的方法，即利用 `RoI pooling` 或 `RoI align` 提取预测框的内部信息，从而获得感知能力。但这样做开销很大，因此我们提出了用关键点三元组来检测目标，这样使得我们的方法在 one-stage 的前提下就能获得感知物体内部信息的能力。并且开销较小，因为我们只需关注物体的中心，从而避免了 `RoI pooling` 或 `RoI align` 关注物体内部的全部信息。



### 方法介绍

#### 1. 利用关键点三元组检测物体

![](/img/CenterNet/post-CenterNet-fig2.png)

图为 `CenterNet` 的结构图。网络通过 `center pooling` 和 `cascade corner pooling` 分别得到 `center heatmap` 和 `corner heatmaps`，用来预测关键点的位置。得到角点的位置和类别后，通过 `offsets` 将角点的位置映射到输入图片的对应位置，然后通过 `embedings` 判断哪两个角点属于同一个物体，以便组成一个检测框。正如前文所说，组合过程中由于缺乏来自目标区域内部信息的辅助，从而导致大量的误检。为了解决这一问题，`CenterNet` 不仅预测角点，还预测中心点。我们对每个预测框定义一个中心区域，通过判断每个目标框的中心区域是否含有中心点，若有则保留，并且此时框的 confidence 为中心点，左上角点和右下角点的confidence的平均，若无则去除，使得网络具备感知目标区域内部信息的能力，能够有效除错误的目标框。

我们发现中心区域的尺度会影响错误框去除效果。中心区域过小导致很多小尺度的错误目标框无法被去除，而中心区域过大导致很多大尺度的错误目标框无法被去除，因此我们提出了尺度可调节的中心区域定义法 (公式)。该方法可以在预测框的尺度较大时定义一个相对较小的中心区域，在预测框的尺度较小时预测一个相对较大的中心区域。如 Fig3 所示。

![](/img/CenterNet/post-CenterNet-fm1.png)

![](/img/CenterNet/post-CenterNet-fig3.png)

#### 2. 提取中心点和角点

![](/img/CenterNet/post-CenterNet-fig4.png)

**Center Pooling：**一个物体的中心并不一定含有很强的，易于区分其他类别的语义信息。例如，一个人的头部含有很强的，易于区分其他类别的语义信息，但其几何中心往往位于人的中部。本文提出了`Center Pooling`来丰富中心点特征。`Center Pooling`提取中心点水平方向和垂直方向的最大值并相加，以此给中心点提供所处位置以外的信息。这一操作，使得中心点有机会获得更易于区分其他类别的语义信息。`center pooling`可通过不同方向上的`corner pooling`组合来实现。一个水平方向上的取最大值操作，可以由`left pooling`和`right pooling`通过串联实现，同理，一个垂直方向上的取最大值操作可由`top pooling`和`bottom pooling`通过串联实现。

![](/img/CenterNet/post-CenterNet-fig5.png)

**Cascade Corner Pooling**：一般情况下角点位于物体外部，所处位置并不含有关联物体的语义信息，这为角点的检测带来了困难。下图为传统做法，称为`corner pooling`，它提取物体边界最大值并相加，该方法只能提供关联物体边缘语义信息，对于更加丰富的物体内部语义信息则很难提取到。

![](/img/CenterNet/post-CenterNet-fig6.png)

下图是`Cascade Corner Pooling`原理，它首先提取物体边界最大值，然后在边界最大值处继续向内部提取最大值，并与边界最大值相加。以此给角点特征提供更加丰富的关联物体语义信息。`Cascade Corner Pooling`也可以通过不同方向上的`Corner Pooling`的组合实现。

![](/img/CenterNet/post-CenterNet-fig7.png)

### 实验分析

![](/img/CenterNet/post-CenterNet-result1.png)

本实验在最具挑战性之一的 MS COCO 数据集上进行测试， 我们选了一些比较有代表性的工作做了对比。实验结果表明 CenterNet 获得了47%的AP，超过了所有已知的 one-stage 检测方法，并大幅度领先，其领先幅度至少达4.9%。上表为 CenterNet 与 CornerNet 的单独对比。最近目标检测方法在COCO数据集上基本在以百分之零点几的精度往前推进，因为coco数据集难度很高，而我们的 CenterNet 往前推进了将近5个百分点。同时，CenterNet 的结果也接近two-stage方法的最好结果。值得注意的是，CenterNet 训练输入图片分辨率只有 511X511，在 single-scale下，测试图片的分辨率为原图分辨率（~500），在 multi-scale下，测试图片的分辨率最大为原图分辨率的1.8倍。而two-stage的输入图片的分辨率一般最短边也要>600,甚至更大，比如D-RFCN+SNIP[^2] 和 PANet[^3]。而且我们的方法是 Train from scratch。

速度方面，Two-stage 方法论文中一般是不报的。One-stage方法只在较浅的backbone上如VGG-16上报速度，一般处理一张图片需要十几毫秒，在较深的backbone上速度为慢一些，处理一张图片需要几百毫秒，但还是要比 two-stage 的方法快。在这里，我们在一张 Nvidia Tesla P100 显卡上比较了CornerNet和CenterNet，CornerNet511-104 测试速度约为 300ms/帧 (并没有实现原论文所说的250ms/帧的速度，可能是与我用的服务器环境有关)，而 CenterNet511-104 的测试速度约为340ms/帧，比 baseline 慢约 40ms/帧。但对于更轻backbone，CenterNet511-52的测试速度约为270ms/帧，比CornerNet511-104快约30ms/帧，而且其精度无论是single-scale test 还是 multi-scale test 都比CornerNet511-104高。

![](/img/CenterNet/post-CenterNet-result2.png)

![](/img/CenterNet/post-CenterNet-result3.png)

![](/img/CenterNet/post-CenterNet-result4.png)

上表为消除实验。第一行为 CornerNet 结果。中心点的加入 (CRE) 使得网络提升了2.3% (37.6% vs 39.9%)。对于中心点的检测，本实验使用传统的卷积操作进行。**其中小尺度目标提升的最多**，提升了4.6% (18.5% vs 23.1%)， 而大尺度目标几乎没有发生变化。**这说明小尺度的错误目标框被去除的最多，这是因为从概率上讲，小尺度目标框由于面积小更容易确定其中心点，因此那些错误的小目标框不在中心点附近的概率更大，因此去除的最多。**Center pooling (CTP) 的加入使网络进一步提升了0.9%。值得注意的是，大尺度目标提升了1.4% (52.2% vs 53.6%)，小目标和中等目标也得到了一定的提升，这表明 center pooling 能够使中心点获得更易于区分于其他类别的语义信息。Cascade corner pooling (CCP) 使得使网络性能进一步提升。第二行的试验中，我们将 CornerNet 的corner pooling 替换成了 cascade corner pooling，性能提升了0.7% (37.6% vs 38.3%)。可以观察到大目标的 AP 没有发生变化，AR 却提升了1.8% (74.0% vs 75.8%)， 这说明 cascade corner pooling 通过加入了物体内部信息能够感知更多的物体，但是由于大目标由于面积过大，使其容易获得较明显的内部特征而干扰了边缘特征，因此使得预测出的目标框位置不精确。当结合了 CRE 后，由于 CRE 能够有效去除错误目标框，因此使大目标框的AP得到了提升 (53.6% vs 55.8%).

（这部分的实现结果分析挺不错的，看原论文。。。）



#### 参考文献

[^1]: H. Law and J. Deng. [Cornernet](https://arxiv.org/abs/1904.08189): Detecting objects as paired keypoints. In Proceedings of the European conference on computer vision, pages 734–750, 2018.
[^2]: B. Singh and L. S. Davis. An analysis of scale invariance in object detection snip. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3578–3587, 2018.
[^3]: S. Liu, L. Qi, H. Qin, J. Shi, and J. Jia. Path aggregation network for instance segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8759–8768, 2018.

