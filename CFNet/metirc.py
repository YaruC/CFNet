from medpy import metric
# 计算评估指标
# gt是(4,240,240)且为0,1,2,3
def calculate_metric_percase(pred, gt):
    # 将结果中大于0的元素设置为1，用于二值化真实标签
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    # 检查是够存在正样本
    if pred.sum() > 0 and gt.sum()>0:
        # 二分类
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
