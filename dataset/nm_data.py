"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Dataset(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, dataset_info, root, transform=None):
        self.root = root
        self._gt_img_list, self._gt_lane_list, self._gt_fs_list = \
            self._init_dataset(dataset_info, root)
        self.transform = transform
        self.ids = self._gt_img_list
        # for (year, name) in image_sets:
        #     rootpath = osp.join(self.root, 'VOC' + year)
        #     for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
        #         self.ids.append((rootpath, line.strip()))

    def _init_dataset(self, dataset_info_file, dataset_dir):
        """

        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_lane_list = []
        gt_freespace_list = []

        assert osp.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                gt_img_list.append(dataset_dir + info_tmp[0])
                gt_lane_list.append(dataset_dir + info_tmp[1])
                gt_freespace_list.append(dataset_dir + info_tmp[2])

        return gt_img_list, gt_lane_list, gt_freespace_list

    def __getitem__(self, index):
        im, lane, fs = self.pull_item(index)

        return im, lane, fs

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self._gt_img_list[index]
        lane_id = self._gt_lane_list[index]
        fs_id = self._gt_fs_list[index]
        img = cv2.imread(img_id)
        lane = cv2.imread(lane_id, cv2.IMREAD_GRAYSCALE)
        fs = cv2.imread(fs_id, cv2.IMREAD_GRAYSCALE)
        fs = fs/np.max(fs)
        height, width, channels = img.shape

        if self.transform is not None:
            img, lane, fs = self.transform(img, lane, fs)
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
        #return torch.from_numpy(img).permute(2, 0, 1), lane, fs
        return img, lane, fs
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).
    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations
    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    imgs = []
    lane = []
    fs = []
    for sample in batch:
        imgs.append(torch.from_numpy(sample[0].transpose(2,0,1)))
        lane.append(torch.from_numpy(sample[1]))
        fs.append(torch.from_numpy(sample[2]))
    return torch.stack(imgs, 0), torch.stack(lane, 0).long(), torch.stack(fs, 0).long()
