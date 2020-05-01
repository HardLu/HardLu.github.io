---
layout: post
title: "facebook-maskrcnn-benchmark 框架中踩过的坑(DOTA 数据集)"
subtitle: 'Object Detection structure bugs'
author: "WenlSun"
header-img: "img/post-bugs.jpg"
header-style: text
tags:
  - Object Detection
  - 笔记
  - pytorch
---

### Out of Memory(内存溢出)

#### 1. 由于DOTA数据集中有的一幅图像中包含的目标数量太多，导致在计算bounding box 时内存溢出。

+ 问题位置

  `maskrcnn_benchmark/structures/boxlist_ops.py`文件中的 `boxlist_iou`函数

+ 解决方法

  将一部分的计算移到 `CPU` 进行计算，计算完成之后移回`GPU` ,具体操作如下面代码所示

  原始的`boxlist_iou` 函数

  ```python
  def boxlist_iou(boxlist1, boxlist2):
      """Compute the intersection over union of two set of boxes.
      The box order must be (xmin, ymin, xmax, ymax).
  
      Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].
  
      Returns:
        (tensor) iou, sized [N,M].
  
      Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
      """
      if boxlist1.size != boxlist2.size:
          raise RuntimeError(
                  "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
  
      N = len(boxlist1)
      M = len(boxlist2)
  
      area1 = boxlist1.area()
      area2 = boxlist2.area()
  
      box1, box2 = boxlist1.bbox, boxlist2.bbox
  
      lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
      rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
  
      TO_REMOVE = 1
  
      wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
      inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
  
      iou = inter / (area1[:, None] + area2 - inter)
      return iou
  ```

  将其修改为：

  ```python
  def boxlist_iou(boxlist1, boxlist2):
      """Compute the intersection over union of two set of boxes.
      The box order must be (xmin, ymin, xmax, ymax).
  
      Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].
  
      Returns:
        (tensor) iou, sized [N,M].
  
      Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
      """
      if boxlist1.size != boxlist2.size:
          raise RuntimeError(
                  "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))
  
      N = len(boxlist1)
      M = len(boxlist2)
  
      area1 = boxlist1.area()
      area2 = boxlist2.area()
  
      box1, box2 = boxlist1.bbox, boxlist2.bbox
  	# TODO 这儿做判断，当bbox的数量大于一个数(此处为100)时，将移动到CPU进行计算，否则在GPU上计算
      USE_CPU_MODE = True
      if USE_CPU_MODE and N > 100:
          device = box1.device
          box1 = box1.cpu()
          box2 = box2.cpu()
  
          lt = torch.max(box1[:, None, :2], box2[:, :2]).cpu()  # [N,M,2]
          rb = torch.min(box1[:, None, 2:], box2[:, 2:]).cpu()  # [N,M,2]
  
          TO_REMOVE = 1
  
          wh = (rb - lt + TO_REMOVE).clamp(min=0).cpu()  # [N,M,2]
          inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
  
          iou = inter / (area1[:, None].cpu() + area2.cpu() - inter.cpu())
          iou = iou.to(device)
          return iou
      else:
          lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
          rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
  
          TO_REMOVE = 1
  
          wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
          inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
  
          iou = inter / (area1[:, None] + area2 - inter)
          return iou
  ```
#### 2. 由于DOTA数据集中有的一幅图像中包含的目标数量太多，导致在与ground truth进行匹配的时候内存溢出。

+ 问题位置

  `maskrcnn_benchmark/modeling/matcher.py` 文件中的 `set_low_quality_matches_`函数

+ 解决方法

  将一部分的计算移到 `CPU` 进行计算，计算完成之后移回`GPU` ,具体操作如下面代码所示

  原始的`boxlist_iou` 函数

原始的`set_low_quality_matches_`:

```python
def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
```

将其修改为：

```python
def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        N = match_quality_matrix.size()[0]
        USE_CPU_MODE = True
        if USE_CPU_MODE and N > 1000: # 这儿设置阈值
            device = match_quality_matrix.device
            match_quality_matrix = match_quality_matrix.cpu()
            highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
            # Find highest quality match available, even if it is low, including ties
            gt_pred_pairs_of_highest_quality = torch.nonzero(
                match_quality_matrix == highest_quality_foreach_gt[:, None]
            )
            gt_pred_pairs_of_highest_quality.to(device)
        else:
            highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
            # Find highest quality match available, even if it is low, including ties
            gt_pred_pairs_of_highest_quality = torch.nonzero(
                match_quality_matrix == highest_quality_foreach_gt[:, None])

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]
```
### 输入图像通道问题

+ `maskrcnn_benchmark` 训练时输入到网络的图像通道是`BGR` (即cv2 读取图像的模式)，在预测时，需要注意读取的时候需要将读取的图像转换成`BGR` 格式，否则预测出来的结果惨不忍睹(尤其是游泳池这类)。



其他 bug 尚在挖掘中。。。。