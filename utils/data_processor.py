#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-11 下午4:58
# @Author  : Luo Yao
# @Site    : http://icode.baidu.com/repos/baidu/personal-code/Luoyao
# @File    : lanenet_data_processor.py
# @IDE: PyCharm Community Edition
"""
实现LaneNet的数据解析类
"""
import os.path as ops

import cv2
import numpy as np

from PIL import Image
import os
from matplotlib import pyplot as plt

VGG_MEAN = [123, 117, 104]


class DataSet(object):
    """
    实现数据集类
    """

    def __init__(self, dataset_info_file, dataset_dir):
        """

        :param dataset_info_file:
        """
        self._gt_img_list, self._gt_label_binary_list, \
        self._gt_label_instance_list = self._init_dataset(dataset_info_file, dataset_dir)
        # self._random_dataset()
        self._next_batch_loop_count = 0

    def _init_dataset(self, dataset_info_file, dataset_dir):
        """

        :param dataset_info_file:
        :return:
        """
        gt_img_list = []
        gt_label_binary_list = []
        gt_label_instance_list = []

        assert ops.exists(dataset_info_file), '{:s}　不存在'.format(dataset_info_file)

        with open(dataset_info_file, 'r') as file:
            for _info in file:
                info_tmp = _info.strip(' ').split()
                # for CULane/small_list
                # gt_img_list.append(ops.join(dataset_dir, '..') + info_tmp[0])
                # gt_label_binary_list.append(ops.join(dataset_dir, '..') + info_tmp[1])
                # gt_label_instance_list.append(ops.join(dataset_dir, '..') + info_tmp[1])
                gt_img_list.append(dataset_dir + info_tmp[0])
                gt_label_binary_list.append(dataset_dir + info_tmp[1])
                gt_label_instance_list.append(dataset_dir + info_tmp[1])

        return gt_img_list, gt_label_binary_list, gt_label_instance_list

    def _random_dataset(self):
        """

        :return:
        """
        assert len(self._gt_img_list) == len(self._gt_label_binary_list) == len(self._gt_label_instance_list)

        random_idx = np.random.permutation(len(self._gt_img_list))
        new_gt_img_list = []
        new_gt_label_binary_list = []
        new_gt_label_instance_list = []

        for index in random_idx:
            new_gt_img_list.append(self._gt_img_list[index])
            new_gt_label_binary_list.append(self._gt_label_binary_list[index])
            new_gt_label_instance_list.append(self._gt_label_instance_list[index])

        self._gt_img_list = new_gt_img_list
        self._gt_label_binary_list = new_gt_label_binary_list
        self._gt_label_instance_list = new_gt_label_instance_list

    def next_batch(self, batch_size):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size

        if idx_end > len(self._gt_label_binary_list):
            self._random_dataset()
            self._next_batch_loop_count = 0
            return self.next_batch(batch_size)
        else:
            gt_img_list = self._gt_img_list[idx_start:idx_end]
            gt_label_binary_list = self._gt_label_binary_list[idx_start:idx_end]
            gt_label_instance_list = self._gt_label_instance_list[idx_start:idx_end]

            gt_imgs = []
            gt_labels_binary = []
            gt_labels_instance = []

            for gt_img_path in gt_img_list:
                # print(gt_img_path)
                gt_imgs.append(Image.open(gt_img_path))

            for gt_label_path in gt_label_instance_list:
                label_img = Image.open(gt_label_path)
                if np.max(label_img) != 2:
                    print(str(np.max(label_img))+gt_img_path)
                gt_labels_instance.append(label_img)

            # for gt_label_path in gt_label_binary_list:
            #     label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
            #     label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
            #     idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
            #
            #     label_binary[idx] = 1
            #     if np.max(label_binary) != 1:
            #         print(str(np.max(label_binary))+gt_img_path)
            #     gt_labels_binary.append(label_binary)
            #


            self._next_batch_loop_count += 1
            return gt_imgs, gt_labels_instance

    def next_item(self, batch_size=1):
        """

        :param batch_size:
        :return:
        """
        assert len(self._gt_label_binary_list) == len(self._gt_label_instance_list) \
               == len(self._gt_img_list)

        idx_start = batch_size * self._next_batch_loop_count
        idx_end = batch_size * self._next_batch_loop_count + batch_size


        gt_img_list = self._gt_img_list[idx_start:idx_end]
        gt_label_binary_list = self._gt_label_binary_list[idx_start:idx_end]
        gt_label_instance_list = self._gt_label_instance_list[idx_start:idx_end]

        gt_imgs = []
        gt_labels_binary = []
        gt_labels_instance = []

        for gt_img_path in gt_img_list:
            # print(gt_img_path)
            gt_imgs.append(Image.open(gt_img_path))

        for gt_label_path in gt_label_instance_list:
            label_img = Image.open(gt_label_path)
            if np.max(label_img) != 2:
                print(str(np.max(label_img))+gt_img_path)
            gt_labels_instance.append(label_img)

        # for gt_label_path in gt_label_binary_list:
        #     label_img = cv2.imread(gt_label_path, cv2.IMREAD_COLOR)
        #     label_binary = np.zeros([label_img.shape[0], label_img.shape[1]], dtype=np.uint8)
        #     idx = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        #
        #     label_binary[idx] = 1
        #     if np.max(label_binary) != 1:
        #         print(str(np.max(label_binary))+gt_img_path)
        #     gt_labels_binary.append(label_binary)



        self._next_batch_loop_count += 1
        if self._next_batch_loop_count == len(self._gt_label_binary_list):
            #self._random_dataset()
            self._next_batch_loop_count = 0

        return gt_imgs, gt_labels_instance


def draw_res(X, y, pre, epoch, path = 'mid_result/'):
    if not ops.exists(path):
        os.makedirs(path)
    X = X.data.cpu().numpy()
    y = y.data.cpu().numpy()
    pre =pre.data.cpu().numpy()
    for i in range(len(X)):
        fig = plt.figure()
        ax1 = fig.add_subplot(131)
        img = X[i].transpose(1, 2, 0).astype(int) + VGG_MEAN
        plt.imshow(img)
        label = y[i]
        ax2 = fig.add_subplot(132)
        plt.imshow(label)
        ax3 = fig.add_subplot(133)
        pred = pre[i][1]
        plt.imshow(pred)
        plt.savefig(path+'{}_{}.jpg'.format(epoch, i))
    print('save_midresult:{}'.format(epoch))
        #cv2.imwrite('mid_result/{}_{}.jpg'.format(epoch, i), img)

def draw_fs(X, y1, y2, pre1, pre2, epoch, path = 'mid_result/'):
    if not ops.exists(path):
        os.makedirs(path)
    X = X.data.cpu().numpy()
    y1 = y1.data.cpu().numpy()
    pre1 =pre1.data.cpu().numpy()
    y2 = y2.data.cpu().numpy()
    pre2 = pre2.data.cpu().numpy()
    for i in range(len(X)):
        fig = plt.figure()
        ax1 = fig.add_subplot(231)
        img = X[i].transpose(1, 2, 0).astype(int) + VGG_MEAN
        plt.imshow(img)
        label = y1[i]
        ax2 = fig.add_subplot(232)
        plt.imshow(label)
        ax3 = fig.add_subplot(233)
        pred1 = pre1[i][1]
        plt.imshow(pred1)
        ax4 = fig.add_subplot(234)
        plt.imshow(y2[i])
        ax5 = fig.add_subplot(235)
        pred2 = pre2[i][1]
        plt.imshow(pred2)
        plt.savefig(path+'{}_{}.jpg'.format(epoch, i))
    print('save_midresult:{}'.format(epoch))
        #cv2.imwrite('mid_result/{}_{}.jpg'.format(epoch, i), img)

if __name__ == '__main__':
    val = DataSet('/home/baidu/DataBase/Semantic_Segmentation/Kitti_Vision/data_road/lanenet_training/train.txt')
    a1, a2, a3 = val.next_batch(1)
    cv2.imwrite('test_binary_label.png', a2[0] * 255)
    b1, b2, b3 = val.next_batch(50)
    c1, c2, c3 = val.next_batch(50)
    dd, d2, d3 = val.next_batch(50)
