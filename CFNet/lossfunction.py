import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn.functional as F
import torch.nn as nn
import SimpleITK as sitk
import copy
from PIL import Image
import os
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    # 输入1,224,224
    # 收一个输入张量 input_tensor，并返回一个经过独热编码处理的输出张量
    def _one_hot_encoder(self, input_tensor):
        tensor_list = [] #存储每个类别的临时独热编码张量
        # ground_truth中的标签为0,1,2,4
        for i in range(self.n_classes):
            # if i == 3:
            #     i = 4  #计算label为4时的编码，而没有label为3时候的编码
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)   #将输入张量中等于i的位置置为True，其余位置为False
            tensor_list.append(temp_prob.unsqueeze(1))  #加一个维度
        output_tensor = torch.cat(tensor_list, dim=1)   #沿着维度拼接起来形成独热编码
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    # input（1,3,224,224）.eXX   target（1,224,224）整形
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:  #true
            # inputs变成0.XX的概率
            inputs = torch.softmax(inputs, dim=1)    #1,3,224,224
        #label的编码0,1,2,4
        target = self._one_hot_encoder(target)   #1,224,224-》1,3,224,224 （变成1,0代码，有不同的通道）
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes): #i为0,1,2,3
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

class FocalLoss(nn.Module):
    def __init__(self,n_classes, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.n_classes = n_classes
    def _one_hot_encoder(self, input_tensor):
        tensor_list = [] #存储每个类别的临时独热编码张量
        # ground_truth中的标签为0,1,2,4
        for i in range(self.n_classes):
            # if i == 3:
            #     i = 4
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)   #将输入张量中等于i的位置置为True，其余位置为False
            tensor_list.append(temp_prob.unsqueeze(1))  #加一个维度
        output_tensor = torch.cat(tensor_list, dim=1)   #沿着维度拼接起来形成独热编码
        return output_tensor.float()

    def forward(self, inputs, targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = self._one_hot_encoder(targets)

        pt = (probs * targets_onehot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma

        alpha_weight = self.alpha * targets_onehot + (1 - self.alpha) * (1 - targets_onehot)


        expanded_focal_weight = focal_weight.unsqueeze(1)  # 在第1维度上添加一个维度
        expanded_focal_weight = expanded_focal_weight.expand_as(log_probs)  # 将维度扩展为与log_probs相同


        loss = -alpha_weight * expanded_focal_weight * log_probs
        loss = loss.mean()

        return loss

# 计算评估指标
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

def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    label =  label.squeeze(0).cpu().detach().numpy()
    image_t1,image_t2 = image[0].squeeze(0).cpu().detach().numpy(),image[1].squeeze(0).cpu().detach().numpy()
    image_t1ce,image_flair = image[2].squeeze(0).cpu().detach().numpy(),image[3].squeeze(0).cpu().detach().numpy()
    if len(image_t1.shape) == 3:
        # 创建空值，保存结果
        prediction = np.zeros_like(label)
        # 从输入中得到所有图片的切片，把那个分成patch
        for ind in range(image_t1.shape[0]): #范围到147
            slice_t1 = image_t1[ind, :, :]  #得到一个通道里的1片（147个通道）
            slice_t2 = image_t2[ind, :, :]
            slice_t1ce = image_t1ce[ind, :, :]
            slice_flair = image_flair[ind, :, :]
            x_t1, y_t1 = slice_t1.shape[0], slice_t1.shape[1]  #512,512
            if x_t1 != patch_size[0] or y_t1 != patch_size[1]:
                slice_t1 = zoom(slice_t1, (patch_size[0] / x_t1, patch_size[1] / y_t1), order=3)  # previous using 0

            x_t2, y_t2 = slice_t2.shape[0], slice_t2.shape[1]  # 512,512
            if x_t2 != patch_size[0] or y_t2 != patch_size[1]:
                slice_t2 = zoom(slice_t2, (patch_size[0] / x_t2, patch_size[1] / y_t2), order=3)

            x_t1ce, y_t1ce = slice_t1ce.shape[0], slice_t1ce.shape[1]  # 512,512
            if x_t1ce != patch_size[0] or y_t1ce != patch_size[1]:
                slice_t1ce = zoom(slice_t1ce, (patch_size[0] / x_t1ce, patch_size[1] / y_t1ce), order=3)

            x_flair, y_flair = slice_flair.shape[0], slice_flair.shape[1]  # 512,512
            if x_flair != patch_size[0] or y_flair != patch_size[1]:
                slice_flair = zoom(slice_flair, (patch_size[0] / x_flair, patch_size[1] / y_flair), order=3)

            input_t1 = torch.from_numpy(slice_t1).unsqueeze(0).unsqueeze(0).float().cuda() #1,1,224,224
            input_t2 = torch.from_numpy(slice_t2).unsqueeze(0).unsqueeze(0).float().cuda() #1,1,224,224
            input_t1ce = torch.from_numpy(slice_t1ce).unsqueeze(0).unsqueeze(0).float().cuda() #1,1,224,224
            input_flair = torch.from_numpy(slice_flair).unsqueeze(0).unsqueeze(0).float().cuda() #1,1,224,224

            input = [input_t1,input_t2,input_t1ce,input_flair]
            # 将两个片结合起来输入
            net.eval()
            with torch.no_grad():
        #没有相应的权重，暂时测不出来
                # 得到网络输出,输入是(1,1,224,224)输出是(1,9,224,224)
                #输入进去肯定是所有数据，输出的是一个结果
                outputs = net(input)
                # 此处得到的就是0,1,2,3
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0) #去除维度为1的，224,224
                # out[out == 3] = 4  #不能直接将3改成4
                out = out.cpu().detach().numpy()  #224,224
                if x_t1 != patch_size[0] or y_t1 != patch_size[1]:
                    pred = zoom(out, (x_t1 / patch_size[0], y_t1 / patch_size[1]), order=0)  #512,512
                else:
                    pred = out
                prediction[ind] = pred   #将所有片的预测复制连接起来，一个片（512,512），复制到第一个维度的147个通道上，也就有147个循环

    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            # 得到网络输出
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            # 从GPU上转移到cpu上得到预测值
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):  #i 1,2,3
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 进行对比什么的是不改变分类种类颜色的，只有全都弄完了，进行保存的时候改变
    if test_save_path is not None:
        # 转换为SimpleITK图像
        image_t1_itk = sitk.GetImageFromArray(image_t1.astype(np.float32))
        image_t2_itk = sitk.GetImageFromArray(image_t2.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        image_t1_itk.SetSpacing((1, 1, z_spacing))
        image_t2_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        # 保存为NIFTI格式的文件（得到的groundtruth不对）
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(image_t1_itk, test_save_path + '/'+ case + "_t1_img.nii.gz")
        sitk.WriteImage(image_t2_itk, test_save_path + '/'+ case + "_t2_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    # 保存了测试结果
    return metric_list


"""
def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    _,x, y = image.shape
    if x != patch_size[0] or y != patch_size[1]:
        #缩放图像符合网络输入
        image = zoom(image, (1,patch_size[0] / x, patch_size[1] / y), order=3)
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        # 得到网络的输出
        out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()
        if x != patch_size[0] or y != patch_size[1]:
            #缩放图像至原始大小
            prediction = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
        else:
            prediction = out

    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        #深度拷贝
        a1 = copy.deepcopy(prediction)
        a2 = copy.deepcopy(prediction)
        a3 = copy.deepcopy(prediction)
#表示有4个类别
        a1[a1 == 1] = 255
        a1[a1 == 2] = 0   #代表R通道中输出结果为2的赋值0
        a1[a1 == 3] = 255
        a1[a1 == 4] = 20

        a2[a2 == 1] = 255
        a2[a2 == 2] = 255 #代表G通道中输出结果为2的赋值255
        a2[a2 == 3] = 0
        a2[a2 == 4] = 10

        a3[a3 == 1] = 255
        a3[a3 == 2] = 77 #代表B通道中输出结果为2的赋值77;(0,255,77,)对应就是绿色，类别2就是绿色
        a3[a3 == 3] = 0
        a3[a3 == 4] = 120

        a1 = Image.fromarray(np.uint8(a1)).convert('L') #array转换成image，Image.fromarray(np.uint8(img))
        a2 = Image.fromarray(np.uint8(a2)).convert('L')
        a3 = Image.fromarray(np.uint8(a3)).convert('L')
        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(test_save_path+'/'+case+'.png')


    return metric_list



def test_single_volume(image, label, net, classes, patch_size=[240, 240], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    # 判断输入是否是RGB格式
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            # 检查当前切片的尺寸是否与指定的块大小（patch_size）不匹配
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                # 得到网络输出
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            # 得到网络输出
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            # 从GPU上转移到cpu上得到预测值(和label)
            prediction = out.cpu().detach().numpy()

    metric_list = []
    for i in range(1, classes):
        # 计算度量指标
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    # 如何将其保存为一般的图像？
    if test_save_path is not None:
        # 深度拷贝
        a1 = copy.deepcopy(out)
        a2 = copy.deepcopy(out)
        a3 = copy.deepcopy(out)

        # 表示有4个类别
        a1[a1 == 0] = 0
        a1[a1 == 1] = 255
        a1[a1 == 2] = 0  # 代表R通道中输出结果为2的赋值0
        a1[a1 == 3] = 255
        a1[a1 == 4] = 128
        a1[a1 == 5] = 138
        a1[a1 == 6] = 147
        a1[a1 == 7] = 0
        a1[a1 == 8] = 0


        a2[a2 == 0] = 0
        a2[a2 == 1] = 0
        a2[a2 == 2] = 255  # 代表G通道中输出结果为2的赋值255
        a2[a2 == 3] = 0
        a2[a2 == 4] = 0
        a2[a2 == 5] = 43
        a2[a2 == 6] = 112
        a2[a2 == 7] = 0
        a2[a2 == 8] = 0

        a3[a3 == 0] = 0
        a3[a3 == 1] = 255
        a3[a3 == 2] = 77  # 代表B通道中输出结果为2的赋值77;(0,255,77,)对应就是绿色，类别2就是绿色
        a3[a3 == 3] = 0
        a3[a3 == 4] = 128
        a3[a3 == 5] = 226
        a3[a3 == 6] = 219
        a3[a3 == 7] = 139
        a3[a3 == 8] = 128

        a1 = a1.astype(np.uint8)
        a2 = a2.astype(np.uint8)
        a3 = a3.astype(np.uint8)

        # 当保存为灰度图时，为(W,H)
        # a1 = torch.squeeze(a1)  # 移除维度为 1 的维度
        # a2 = torch.squeeze(a2)
        # a3 = torch.squeeze(a3)
        a1 = Image.fromarray(a1).convert('L')
        a2 = Image.fromarray(a2).convert('L')
        a3 = Image.fromarray(a3).convert('L')

        # a1 = Image.fromarray(np.uint8(a1)).convert('L')  # array转换成image，Image.fromarray(np.uint8(img))
        # a2 = Image.fromarray(np.uint8(a2)).convert('L')
        # a3 = Image.fromarray(np.uint8(a3)).convert('L')

        prediction = Image.merge('RGB', [a1, a2, a3])
        prediction.save(os.path.join(test_save_path + '/' + case + '.png'))

    return metric_list
"""