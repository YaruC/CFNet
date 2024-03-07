# coding=gbk
import os, argparse, time, datetime, stat, shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import cv2
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from util.MF_dataset import MF_dataset
from util.dataset_synapse import *
from util.augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
from util.util import compute_results
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from lossfunction import DiceLoss, FocalLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms


from model import SFAFMA

# from model import losses


#############################################################################################
parser = argparse.ArgumentParser(description='Train with pytorch')
#############################################################################################
parser.add_argument('--model_name', '-m', type=str, default='SFAFMA')
parser.add_argument('--batch_size', '-b', type=int, default=12)
parser.add_argument('--lr_start', '-ls', type=float, default=0.01)
parser.add_argument('--gpu', '-g', type=int, default=2)
parser.add_argument('--img_height', '-ih', type=int, default=240)
parser.add_argument('--img_width', '-iw', type=int, default=240)
parser.add_argument('--img_size', type=int,
                    default=240, help='input patch size of network input')
parser.add_argument('--volume_path_train', type=str,
                    default='/icislab/volume1/cx/BraTS-5Slices-mater/BraTS-5Slices-master/result/20/',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--volume_path_test', type=str,
                    default='/icislab/volume1/cx/BraTS-5Slices-mater/BraTS-5Slices-master/result/20/',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--volume_path_val', type=str,
                    default='/icislab/volume1/cx/BraTS-5Slices-mater/BraTS-5Slices-master/result/20/',
                    help='root dir for validation volume data')  # for acdc volume_path=root_dir
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--beta', default=0, type=float, help='hyperpar0ameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float, help='cutmix probability')
parser.add_argument('--lr_decay', '-ld', type=float, default=0.96)
parser.add_argument('--epoch_max', '-em', type=int, default=200)  # please stop training mannully
parser.add_argument('--epoch_from', '-ef', type=int, default=0)
parser.add_argument('--num_workers', '-j', type=int, default=6)
parser.add_argument('--n_class', '-nc', type=int, default=4)
parser.add_argument('--list_dir', type=str,
                    default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/SFAF-MA-main/lists/lists_synapse/',
                    help='list dir')
# parser.add_argument('--data_dir_train', '-drtt', type=str,
#                     default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/dataset/19/')
# parser.add_argument('--data_dir_test', '-drte', type=str,
#                     default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/dataset/train/18train/18train/')
# parser.add_argument('--data_dir_val', '-drv', type=str,
#                     default='/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/dataset/train/18train/18train/')
args = parser.parse_args()
#############################################################################################

augmentation_methods = [
    RandomFlip(prob=0.5),
    RandomCrop(crop_rate=0.1, prob=1.0),
]


# 用于设置多线程环境中的随机种子的常见做法，以确保每个工作线程都有独立的随机数生成器状态。
def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train(epo, model, train_loader, optimizer):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    model.train()
    for i_batch, sampled_batch in enumerate(train_loader):
        image_t1, image_t2, image_t1ce, image_flair = sampled_batch['image_t1'], sampled_batch['image_t2'], \
                                                      sampled_batch['image_t1ce'], sampled_batch['image_flair']
        image_batch = [image_t1, image_t2,image_t1ce,image_flair]
        image_batch = [tensor.cuda() for tensor in image_batch]
        names = sampled_batch['case_name'][0]

        label = sampled_batch['label']
        label = label.cuda()
        label[label == 4] = 3
        label = label.squeeze(dim=1)
        # start_t = time.time()
        starter.record()
        # 输入的值是整个数据，得到结果（1,9,240,240）
        # logits= model(image_batch)  # logits.size(): mini_batch*num_class*480*640
        logits= model(image_batch)  # logits.size(): mini_batch*num_class*480*640
        # logits, x_1, x_2 = model(image_batch)  # logits.size(): mini_batch*num_class*480*640
        loss_dice = dice_loss(logits, label[:].long(), softmax=True)
        loss_focal = focal_loss(logits, label[:].long())
        # loss_dice_1 = dice_loss(x_1, label[:].long(), softmax=True)
        # loss_focal_1 = focal_loss(x_1, label[:].long())
        # loss_dice_2 = dice_loss(x_2, label[:].long(), softmax=True)
        # loss_focal_2 = focal_loss(x_2, label[:].long())

        loss = 0.6 * loss_dice + 0.4 * loss_focal
        # loss2 = 0.7 * loss_dice_1 + 0.3 * loss_focal_1
        # loss3 = 0.7 * loss_dice_2 + 0.3 * loss_focal_2
        # loss = loss + loss2 + loss3
        """
        out,pred= model(images)
        loss1_classification = F.cross_entropy(pred[4], labels)
        loss2_segmentation = losses.get_seg_loss(out, labels)
        #loss3_SR_L = losses.get_class_loss(pred[0], pred[1:])

        loss = 0.2*loss1_classification+0.8*loss2_segmentation

        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        """
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_this_epo = 0
        for param_group in optimizer.param_groups:
            lr_this_epo = param_group['lr']
        # 每秒钟处理的元素的数量img/sec
        # print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f, %.2f img/sec, loss %.4f, time %s' \
        #       % (args.model_name, epo, args.epoch_max, i_batch + 1, len(train_loader), lr_this_epo,
        #          len(names) / (time.time() - start_t), float(loss),
        #          datetime.datetime.now().replace(microsecond=0) - start_datetime))
        # print('Train: %s, epo %s/%s, iter %s/%s, lr %.8f img/sec, loss %.4f' \
        #     % (args.model_name, epo, args.epoch_max, i_batch + 1, len(train_loader), lr_this_epo, float(loss)))
        print(
            'Train: {0}, epo {1}/{2}, iter {3}/{4}, lr {5:.5f} img/sec, loss {6:.5f}, loss_dice{7:.5f}, loss_float{8:.5f}'.format(
                args.model_name, epo, args.epoch_max, i_batch + 1, len(train_loader), float(lr_this_epo), float(loss),
                float(loss_dice), float(loss_focal)))


def validation(epo, model, val_loader):
    model.eval()
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            image_t1, image_t2, image_t1ce, image_flair = sampled_batch['image_t1'], sampled_batch['image_t2'], \
                                                          sampled_batch['image_t1ce'], sampled_batch['image_flair']
            image_batch = [image_t1, image_t2, image_t1ce, image_flair]
            image_batch = [tensor.cuda() for tensor in image_batch]
            names = sampled_batch['case_name'][0]
            label = sampled_batch['label']
            label = label.cuda()
            label[label == 4] = 3  # 缺失改变了
            label = label.squeeze(dim=1)
            # 输入的值是整个数据，得到结果（1,9,240,240）
            # start_t = time.time()
            """
            out, pred = model(images)
            loss1_classification = F.cross_entropy(pred[3], labels)
            loss2_segmentation = losses.get_seg_loss(out, labels)
            #loss3_SR_L = losses.get_class_loss(pred[0], pred[1:])

            loss = 0.2 * loss1_classification + 0.8 * loss2_segmentation
            """
            # logits= model(image_batch)  # logits.size(): mini_batch*num_class*480*640
            logits= model(image_batch)  # logits.size(): mini_batch*num_class*480*640
            # logits, x_1, x_2 = model(image_batch)  # logits.size(): mini_batch*num_class*480*640
            loss_dice = dice_loss(logits, label[:].long(), softmax=True)
            loss_focal = focal_loss(logits, label[:].long())
            # loss_dice_1 = dice_loss(x_1, label[:].long(), softmax=True)
            # loss_focal_1 = focal_loss(x_1, label[:].long())
            # loss_dice_2 = dice_loss(x_2, label[:].long(), softmax=True)
            # loss_focal_2 = focal_loss(x_2, label[:].long())
    
            loss = 0.6 * loss_dice + 0.4 * loss_focal
            # loss2 = 0.7 * loss_dice_1 + 0.3 * loss_focal_1
            # loss3 = 0.7 * loss_dice_2 + 0.3 * loss_focal_2
            # loss = loss + loss2 + loss3
            # print('Val: %s, epo %s/%s, iter %s/%s, %.2f img/sec, loss %.4f, time %s' \
            #       % (args.model_name, epo, args.epoch_max, i_batch + 1, len(val_loader), len(names)/(time.time()-start_t), float(loss),
            #         datetime.datetime.now().replace(microsecond=0)-start_datetime))
            print('Val: %s, epo %s/%s, iter %s/%s,  loss %.4f' \
                  % (args.model_name, epo, args.epoch_max, i_batch + 1, len(val_loader), float(loss)))



def testing(epo, model, test_loader):
    model.eval()
    sumvalue = 0.0
    best = 0.0
    best_i = 0
    conf_total = np.zeros((args.n_class, args.n_class))
    testing_results_file = os.path.join(weight_dir, 'testing_results_file.txt')
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_loader):
            image_t1, image_t2, image_t1ce, image_flair = sampled_batch['image_t1'], sampled_batch['image_t2'], \
                                                          sampled_batch['image_t1ce'], sampled_batch['image_flair']
            image_batch = [image_t1, image_t2,image_t1ce,image_flair]
            image_batch = [tensor.cuda() for tensor in image_batch]
            names = sampled_batch['case_name'][0]
            label = sampled_batch['label']
            label = label.cuda()
            label[label == 4] = 3  # 缺失改变了
            label = label.squeeze(dim=1)
            #logits,x_1,x_2 = model(image_batch)
            logits = model(image_batch)

            label = label.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(
                1).cpu().numpy().squeeze().flatten()  # prediction and label are both 1-d array, size: minibatch*640*480
            conf = confusion_matrix(y_true=label, y_pred=prediction, labels=[0, 1, 2,
                                                                             3])  # conf is args.n_class*args.n_class matrix, vertical axis: groundtruth, horizontal axis: prediction
            conf_total += conf

            print('Test: %s, epo %s/%s, iter %s/%s' % (
            args.model_name, epo, args.epoch_max, i_batch + 1, len(test_loader)))
    precision, recall, IoU, f1_score = compute_results(conf_total)
    if epo == 0:
        with open(testing_results_file, 'w') as f:
            f.write("# %s, initial lr: %s, batch size: %s, date: %s \n" %(args.model_name, args.lr_start, args.batch_size, datetime.date.today()))
            f.write(
                "# epoch: unlabeled, necrotic tumor core    , peritumoral edema    , enhancing tumor  , average(nan_to_num). (Acc %, IoU %)\n")
    with open(testing_results_file, 'a') as f:
        f.write(str(epo) + ': ')
        # 每一轮下的值
        for i in range(len(precision)):
            f.write('%0.4f, %0.4f, ' % (100 * recall[i], 100 * IoU[i]))
            f.write('结束')
            if not np.isnan(f1_score[i]):
                sumvalue =  100 * np.mean(np.nan_to_num(IoU))
            if sumvalue > best:
                best = sumvalue
                best_i = i
        # 所有轮的平均值;某一轮的平均最好就是最好
        f.write('所有类平均: %0.4f, %0.4f\n' % (100 * np.mean(np.nan_to_num(recall)), 100 * np.mean(np.nan_to_num(IoU))))
        f.write('最好的结果在epoch:{0}:'.format(best_i))

    print("\n* average values (np.mean(x)): \n recall: %.6f, iou: %.6f, precision: %.6f, f1score: %.6f " \
          % (recall.mean(), IoU.mean(), precision.mean(), f1_score.mean()))
    # print('\n* the average time cost per frame : %.2f ms, namely, the inference speed is %.2f fps' % (ave_time_cost * 1000 / (len(test_loader) - 5), 1.0 / (ave_time_cost / (len(test_loader) - 5))))
    print('saving testing results.')
    with open(testing_results_file, "r") as file:
        writer.add_text('testing_results', file.read().replace('\n', '  \n'), epo)
    return sumvalue


if __name__ == '__main__':

    torch.cuda.set_device(args.gpu)
    print("\nthe pytorch version:", torch.__version__)
    print("the gpu count:", torch.cuda.device_count())
    print("the current used gpu:", torch.cuda.current_device(), '\n')

    model = eval(args.model_name)(n_class=args.n_class)

    if args.gpu >= 0: model.cuda(args.gpu)

    ce_loss = CrossEntropyLoss()
    focal_loss = FocalLoss(args.n_class)
    dice_loss = DiceLoss(args.n_class)

    optimizer = optim.SGD(model.parameters(), lr=args.lr_start, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay, last_epoch=-1)

    # preparing folders
    if os.path.exists("/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/"):
        shutil.rmtree("/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/")
    weight_dir = os.path.join("/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/", args.model_name)
    os.makedirs(weight_dir)
    os.chmod(weight_dir,
             stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine
    writer = SummaryWriter("/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/tensorboard_log")
    os.chmod("/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/tensorboard_log",
             stat.S_IRWXU)  # allow the folder created by docker read, written, and execuated by local machine
    os.chmod("/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/", stat.S_IRWXU)

    print('training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('weight will be saved in: %s' % weight_dir)

    # 准备数据
    # train_dataset = MF_dataset(data_dir=args.data_dir_train, split='train', transform=augmentation_methods)
    # val_dataset  = MF_dataset(data_dir=args.data_dir_val, split='val')
    # test_dataset = MF_dataset(data_dir=args.data_dir_test, split='test')

    train_dataset = Synapse_dataset(base_dir=args.volume_path_train, list_dir=args.list_dir, split="20train")

    # train_dataset = Synapse_dataset(base_dir=args.volume_path_train, split="train", list_dir=args.list_dir)
    val_dataset = Synapse_dataset(base_dir=args.volume_path_val, split="20val", list_dir=args.list_dir)
    test_dataset = Synapse_dataset(base_dir=args.volume_path_test, split="20test", list_dir=args.list_dir)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    # start_datetime = datetime.datetime.now().replace(microsecond=0)
    accIter = {'train': 0, 'val': 0}
    best_params = None  # 用于保存效果最好的参数
    best_metric = 0.0  # 用于保存当前最佳的指标值，初始化为一个较小的值（适用于准确率等指标，若使用损失函数则应初始化为一个较大的值）
    for epo in range(args.epoch_from, args.epoch_max):
        print('\ntrain %s, epo #%s begin...' % (args.model_name, epo))
        scheduler.step()  # if using pytorch 0.4.1, please put this statement here
        # 训练
        train(epo, model, train_loader, optimizer)
        validation(epo, model, val_loader)

        if epo % 21 == 0:
            checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
            print('saving check point %s: ' % checkpoint_model_file)
            torch.save(model.state_dict(), checkpoint_model_file)
        if epo == args.epoch_max - 1:
            checkpoint_model_file = os.path.join(weight_dir, str(epo) + '.pth')
            print('saving check point %s: ' % checkpoint_model_file)
            torch.save(model.state_dict(), checkpoint_model_file)

        # # 测试
        # testing(epo, model, test_loader)



        # 在每个轮次中训练模型，并得到当前训练轮次的指标值（比如准确率）
        current_metric = testing(epo, model, test_loader)

        # 比较当前指标值与最佳指标值，若当前指标值更好，则更新最佳指标值和最佳参数
        if current_metric > best_metric:
            best_metric = current_metric
            checkpoint_model_file = os.path.join(weight_dir, 'best '+ '.pth')
            print('saving best check point %s: ' % checkpoint_model_file)
            torch.save(model.state_dict(), checkpoint_model_file)
            with open('/icislab/volume1/cx/SFAF-MA-main/SFAF-MA-main/runs/training_results.txt', 'a') as f:
                f.write(f"Epoch:%s \n" % epo)


        scheduler.step()  # if using pytorch 1.1 or above, please put this statement here
    # 保存
    torch.save(model.state_dict(), checkpoint_model_file)