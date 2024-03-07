# coding=gbk
import os, argparse, time, datetime, sys, shutil, stat, torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from util.MF_dataset import MF_dataset, val_dataset
from util.dataset_synapse import *
from util.util import compute_results, visualize, calculate_metric_percase
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from scipy.io import savemat
from model import SFAFMA

# 测试
#############################################################################################
parser = argparse.ArgumentParser(description='Test with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='SFAFMA')
parser.add_argument('--weight_name', '-w', type=str, default='SFAFMA')
parser.add_argument('--file_name', '-f', type=str, default='best .pth')
parser.add_argument('--dataset_split', '-d', type=str, default='test')  # test, test_day, test_night
parser.add_argument('--gpu', '-g', type=int, default=1)
parser.add_argument('--volume_path', type=str,
                    default='/icislab/volume1/cx/BraTS-5Slices-mater/BraTS-5Slices-master/result/20/',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
#############################################################################################
parser.add_argument('--img_height', '-ih', type=int, default=240)
parser.add_argument('--img_width', '-iw', type=int, default=240)
parser.add_argument('--img_size', '-img', type=int, default=240)
parser.add_argument('--num_workers', '-j', type=int, default=16)
parser.add_argument('--list_dir', type=str,
                    default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/SFAF-MA-main/lists/lists_synapse/',
                    help='list dir')
# 默认设置为9类
parser.add_argument('--n_class', '-nc', type=int, default=4)
parser.add_argument('--data_dir', '-dr', type=str, default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/dataset/')
parser.add_argument('--model_dir', '-wd', type=str, default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/')
args = parser.parse_args()
#############################################################################################

if __name__ == '__main__':

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    # prepare save direcotry
    if os.path.exists("./runs") is None:
        # print("previous \"./runs\" folder exist, will delete this folder")
        # shutil.rmtree("./runs")
        os.makedirs("./runs")
        os.chmod("./runs",
                 stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine
    model_dir = os.path.join(args.model_dir, args.weight_name)
    if os.path.exists(model_dir) is False:
        sys.exit("the %s does not exit." % (model_dir))
    model_file = os.path.join(model_dir, args.file_name)
    if os.path.exists(model_file) is True:
        print('use the final model file.')
    else:
        sys.exit('no model file found.')
    print('testing %s: %s on GPU #%d with pytorch' % (args.model_name, args.weight_name, args.gpu))

    #
    conf_total = np.zeros((args.n_class, args.n_class))
    model = eval(args.model_name)(n_class=args.n_class)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
    if args.gpu >= 0: model.cuda(args.gpu)
    print('loading model file %s... ' % model_file)
    pretrained_weight = torch.load(model_file, map_location=lambda storage, loc: storage.cuda(args.gpu))
    own_state = model.state_dict()
    for name, param in pretrained_weight.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print('done!')

    batch_size = 1  # do not change this parameter!

    # 加载数据
    test_dataset = Synapse_dataset(base_dir=args.volume_path, split="20test", list_dir=args.list_dir)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    # worker_init_fn=worker_init_fn)
    ave_time_cost = 0.0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    metric_r = 0.0
    best_v = 0
    model.eval()
    with torch.no_grad():
        # 一提就是一个batch大小的数据
        for i_batch, sampled_batch in enumerate(test_loader):
            image_t1, image_t2, image_t1ce, image_flair = sampled_batch['image_t1'], sampled_batch['image_t2'], \
                                                          sampled_batch['image_t1ce'], sampled_batch['image_flair']
            image_batch = [image_t1, image_t2,image_t1ce, image_flair]
            image_batch = [tensor.cuda() for tensor in image_batch]
            names = sampled_batch['case_name'][0]
            label = sampled_batch['label']
            # 得到的label是0.000,1.000,2.000,3.000
            label = label.cuda()
            
            # 一定是都变成了整数0,1,2,3才能使用这个
            label[label == 4] = 3
            label = label.squeeze(dim=1)
            # label是0,1的
            labels = label
            starter.record()

            # 输入的值是整个数据，得到结果（6,4,240,240），得到的并不是0,1,2,3而是小数1,9e等等;batch_size为6
            logits = model(image_batch)  # logits.size(): mini_batch*num_class*480*640

            # (240,240)0,1,2,3
            label_me = label.squeeze(0).cpu().numpy()
            # (240,240)0,1,2,3
            out = logits.argmax(1).cpu().squeeze().numpy()
            # print("label_me")
            # print(label_me.shape)
            # print(label_me[128])
            # print("out")
            # print(out.shape)
            # print(out[128])

            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            # 转到cpu上
            label = label.cpu().numpy().squeeze().flatten()
            #
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480

            # generate confusion matrix frame-by-frame
            # 混淆矩阵
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2,
                                                                             3])  # conf is an n_class*n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            # (4,4)
            conf_total += conf
            # save demo images
            """
            argmax(1)根据通道维度上的值进行的操作，在4个通道内，返回最大值所在的索引(0,1,2,3)
            一个整体含有batch_size个预测,一个预测在通道上选择最大值作为最终的预测
            """
            visualize(image_name=names, predictions=logits.argmax(1), weight_name=args.weight_name)
            visualize(image_name=names + '_label', predictions=labels, weight_name=args.weight_name)

            # 开始计算dice和HD  （正确的）
            metric_list = []
            flag_out = False
            flag_label = False
            for i in range(1, 4):  # i 1, 2, 3
                metric_list.append(calculate_metric_percase((out == i), (label_me == i)))
            total_sum = 0
            for tup in metric_list:
            # 遍历元组中的每个值，并累加到总和中
                for value in tup:
                    total_sum += value
            if total_sum > best_v:
                best_v = total_sum
                with open('/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/test_results.txt', 'a') as f:
                    f.write(f"name:%s,指标总和:%s \n" % (names,total_sum))
            print('{0}的评价指标：{1}'.format(names,metric_list))
            # # metric_i并没有清0，所以每一次相加转化都有前面的值，所以是不对的
            # metric_i = metric_list
            # metric_r是计算出来的三个类的所有的两个指标
            # 将每一个batch得到的指标转化并给metric_r，且metric_r是不会清零的，所以是三个类的指标累加
            metric_r += np.array(metric_list)

    # list是所有的结果
    print('********************************************')
    metric_r = metric_r / len(test_loader)
    # 没有输出类别为0时的结果
    for i in range(1, 4):  # i为1,2,3
        # 每个类的平均指标
        print('Mean class %d mean_dice %f mean_hd95 %f recall %f' % (i, metric_r[i - 1][0], metric_r[i - 1][1], metric_r[i - 1][2]))

    performance = np.mean(metric_r, axis=0)[0]
    mean_hd95 = np.mean(metric_r, axis=0)[1]
    print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))

    # 计算评估结果
    precision_per_class, recall_per_class, iou_per_class, f1score = compute_results(conf_total)
    conf_total_matfile = os.path.join("./runs", 'conf_' + args.weight_name + '.mat')
    savemat(conf_total_matfile, {'conf': conf_total})  # 'conf' is the variable name when loaded in Matlab

    print('\n###########################################################################')
    print('\n%s: %s test results (with batch size %d) on %s using %s:' % (
    args.model_name, args.weight_name, batch_size, datetime.date.today(), torch.cuda.get_device_name(args.gpu)))
    print('\n* the tested dataset name: %s' % args.dataset_split)
    print('* the tested image count: %d' % len(test_loader))
    print('* the tested image size: %d*%d' % (args.img_height, args.img_width))
    print('* the weight name: %s' % args.weight_name)
    print('* the file name: %s' % args.file_name)
    print(
        "* recall per class: \n    unlabeled: %.6f, necrotic tumor core: %.6f, peritumoral edema: %.6f, enhancing tumor: %.6f" \
        % (recall_per_class[0], recall_per_class[1], recall_per_class[2], recall_per_class[3]))
    print(
        "* iou per class: \n    unlabeled: %.6f, necrotic tumor core: %.6f, peritumoral edema: %.6f, enhancing tumor: %.6f" \
        % (iou_per_class[0], iou_per_class[1], iou_per_class[2], iou_per_class[3]))
    print("* f1score: \n    unlabeled: %.6f, necrotic tumor core: %.6f, peritumoral edema: %.6f, enhancing tumor: %.6f" \
          % (f1score[0], f1score[1], f1score[2], f1score[3]))
    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f, precision: %.6f, f1score: %.6f " \
          % (recall_per_class.mean(), iou_per_class.mean(), precision_per_class.mean(), f1score.mean()))
    print("* average values (np.mean(np.nan_to_num(x))): \n recall: %.6f, iou: %.6f, precision: %.6f, f1score: %.6f " \
          % (
          np.mean(np.nan_to_num(recall_per_class)), np.mean(np.nan_to_num(iou_per_class)), precision_per_class.mean(),
          f1score.mean()))
    # print("dice{0}".format(dice_per_class.mean()))
    print('\n###########################################################################')
