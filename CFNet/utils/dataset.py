import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset

# 没有改变lable数值
def random_rot_flip(image1,image2,image3,image4,label):
    k = np.random.randint(0, 4)
    image1 = np.rot90(image1, k)
    image2 = np.rot90(image2, k)
    image3 = np.rot90(image3, k)
    image4 = np.rot90(image4, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image1 = np.flip(image1, axis=axis).copy()
    image2 = np.flip(image2, axis=axis).copy()
    image3 = np.flip(image3, axis=axis).copy()
    image4 = np.flip(image4, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image1,image2,image3,image4, label   #512,512


def random_rotate(image1,image2,image3,image4,label):
    angle = np.random.randint(-20, 20)
    image1 = ndimage.rotate(image1, angle, order=0, reshape=False)
    image2 = ndimage.rotate(image2, angle, order=0, reshape=False)
    image3 = ndimage.rotate(image3, angle, order=0, reshape=False)
    image4 = ndimage.rotate(image4, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image1,image2,image3,image4, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size
    def __call__(self, sample):
        image_t1,image_t2,image_t1ce,image_flair,label = sample['image_t1'],sample['image_t1'],sample['image_t1ce'],sample['image_flair'],sample['label']

        # 进行旋转和翻转,就是数据增强用的，但是输入要是两个，
        if random.random() > 0.5:
            image_t1,image_t2,image_t1ce,image_flair, label = random_rot_flip(image_t1,image_t2,image_t1ce,image_flair,label)
        elif random.random() > 0.5:
            image_t1,image_t2,image_t1ce,image_flair, label = random_rotate(image_t1,image_t2,image_t1ce,image_flair,label)


        # 所有的都没有变，仍然是float32
        x, y = image_t1.shape  #训练：512,512
        if x != self.output_size[0] or y != self.output_size[1]:
            # image里的数值发生了变化
            image_t1 = zoom(image_t1, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image_t2 = zoom(image_t2, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image_t1ce = zoom(image_t1ce, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            image_flair = zoom(image_flair, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            # 仍然是浮点型，此时的label还是0,1,2,3,4；实际上没看见3
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        # 变成了float32形式
        image_t1 = torch.from_numpy(image_t1.astype(np.float32)).unsqueeze(0)  #训练：1,224,224
        image_t2 = torch.from_numpy(image_t2.astype(np.float32)).unsqueeze(0)  #训练：1,224,224
        image_t1ce = torch.from_numpy(image_t1ce.astype(np.float32)).unsqueeze(0)  #训练：1,224,224
        image_flair = torch.from_numpy(image_flair.astype(np.float32)).unsqueeze(0)  #训练：1,224,224
        # 变成0,1,2,3,4的float32形式
        label = torch.from_numpy(label.astype(np.float32))  #训练：224,224
        # 将label转换成了long类型的（整形）
        sample = {'image_t1': image_t1, 'image_t2': image_t2,'image_t1ce': image_t1ce,'image_flair': image_flair, 'label': label.long()}
        return sample




class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        if self.split == "19train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            # 尝试取出多个模态
            # image, label = data['image_t2'], data['label']
            image_t2, image_t1, image_t1ce, image_flair, label = data['image_t2'], data['image_t1'], data['image_t1ce'], \
                                                                 data['image_flair'], data['label']
            # 自己改动
            image_t1 = np.squeeze(image_t1)
            image_t2 = np.squeeze(image_t2)
            image_t1ce = np.squeeze(image_t1ce)
            image_flair = np.squeeze(image_flair)
            label = np.squeeze(label)
            sample = {'image_t1': image_t1, 'image_t2': image_t2, 'image_t1ce': image_t1ce, 'image_flair': image_flair,
                      'label': label}
        else:
            slice_name = self.sample_list[idx].strip('\n')
            data_path = self.data_dir + "/" + slice_name + '.npz'
            data = np.load(data_path)
            # 尝试取出两个片
            image_t2, image_t1,image_t1ce,image_flair, label = data['image_t2'], data['image_t1'],data['image_t1ce'],data['image_flair'], data['label']
            image_t1 = torch.from_numpy(image_t1.astype(np.float32))
            image_t2 = torch.from_numpy(image_t2.astype(np.float32))
            image_t1ce = torch.from_numpy(image_t1ce.astype(np.float32))
            image_flair = torch.from_numpy(image_flair.astype(np.float32))

            image_t1 = image_t1.permute(2, 0, 1)   #####(1,240,240)
            image_t2 = image_t2.permute(2, 0, 1)   #####(1,240,240)
            image_t1ce = image_t1ce.permute(2, 0, 1)   #####(1,240,240)
            image_flair = image_flair.permute(2, 0, 1)   #####(1,240,240)
            label = torch.from_numpy(label.astype(np.float32))
            label = label.permute(2, 0, 1)
            # 得到的image是各种各样的数值，得到的label是0,1,2,3,4
            sample = {'image_t1': image_t1, 'image_t2': image_t2,'image_t1ce': image_t1ce,'image_flair': image_flair, 'label': label.long()}

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
