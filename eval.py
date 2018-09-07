import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import time
VGG_MEAN = [123.68, 116.779, 103.939]

import os
from optparse import OptionParser

import os.path as ops
from utils import *
from utils import data_processor, lane_postprocess, lane_cluster
from utils.pil_aug import SSDAugmentation
from deeplab import deeplab_vgg16
from dataset.nm_data import Dataset, detection_collate
import torch.utils.data as data
from Instance_loss import Instance_loss

discriminative_loss = Instance_loss()
postprocessor = lane_postprocess.LaneNetPoseProcessor()
cluster = lane_cluster.LaneNetCluster()


def my_eval_net(net, dataset, gpu=False, *args):
    tot = 0
    num_images = len(dataset)
    t1 = time.time()
    data = iter(dataset)
    for i in range(num_images):
        img, lane = next(data)

        Ins_img = lane.numpy()
        Bi_img = 1 * (Ins_img > 0)
        y2 = torch.from_numpy(Bi_img)
        # y1 = torch.unsqueeze(lane, 1)

        X = Variable(img).cuda()
        y1 = Variable(lane).cuda()
        y2 = Variable(y2).cuda()
        t_s = time.time()
        y_pred = net(X)
        t_e = time.time()
        print('infer time: {}'.format(t_e-t_s))
        yp1 = y_pred[:, :2, :, :]
        yp2 = y_pred[:, 2:, :, :]
        prob1 = F.log_softmax(yp1)
        prob2 = F.log_softmax(yp2)
        loss1 = discriminative_loss(prob1, y1)
        loss2 = nn.NLLLoss2d(Variable(torch.FloatTensor([0.4, 1])).cuda())(prob2, y2)
        loss = loss1 + loss2

        binary_seg_images = F.softmax(prob2).data.cpu().numpy()[:, 1, :, :]
        # instance don't use softmax
        instance_seg_images = prob1.data.cpu().numpy()[:, 1, :, :]
        res = -instance_seg_images[0] * (binary_seg_images[0]>0.2)
        res = res / np.max(res) * 255
        cv2.imwrite('mid_result/lane_instance_seg/val/' + 'a_{}.jpg'.format(i), res)
        # for index, binary_seg_image in enumerate(binary_seg_images):
        #     t0 = time.time()
        #     new_img = postprocessor.nms(binary_seg_image)
        #     t1 = time.time()
        #     binary_image = postprocessor.postprocess(binary_seg_image)
        #     t2 = time.time()
        #     mask_image = cluster.get_lane_mask(binary_seg_ret=binary_image,
        #                                        instance_seg_ret=instance_seg_images[index])
        #     t3 = time.time()
        #     new_mask_image = cluster.get_lane_mask(binary_seg_ret=new_img,
        #                                        instance_seg_ret=instance_seg_images[index])
        #     t4 = time.time()
        #     print('nms:{} , postprocess:{} , mask: {} , nms_mask: {}'.format(
        #         t1-t0, t2-t1, t3-t2, t4-t3
        #     ))
        #     # print('cluster time: {}'.format(time.time() - t_start))
        #     fig = plt.figure(figsize=(16, 16))
        #     ax1 = fig.add_subplot(131)
        #     ax1.imshow(img[0].numpy().transpose(1, 2, 0).astype(int) + (104, 117, 123))
        #     ax2 = fig.add_subplot(132)
        #     ax2.imshow(new_mask_image * np.tile((new_img > 0.2)[:, :, np.newaxis], 3))
        #     ax3 = fig.add_subplot(133)
        #     ax3.imshow(mask_image * np.tile((binary_image > 0.2)[:, :, np.newaxis], 3))
        #     fig.tight_layout()
        #     plt.savefig('mid_result/lane_instance_seg/val/' + '{}.jpg'.format(i))
        #     plt.close()
        if i % 10 == 0:
            print(i)
        # if i % 10 == 0:
        #     data_processor.draw_fs(X, y1, y2, F.softmax(prob1), F.softmax(prob2), i,
        #                            'mid_result/{}/val/{}/'.format(args[0], t1))

        tot += loss[0].data

    return tot / num_images


def eval_net(net, dataset, gpu=False):
    tot = 0
    for i, b in enumerate(dataset):
        X = b[0]
        y = b[1]

        X = torch.FloatTensor(X).unsqueeze(0)
        y = torch.LongTensor(y).unsqueeze(0)

        if gpu:
            X = Variable(X, volatile=True).cuda()
            y = Variable(y, volatile=True).cuda()
        else:
            X = Variable(X, volatile=True)
            y = Variable(y, volatile=True)

        y_pred = net(X)

       # y_pred = (F.sigmoid(y_pred) > 0.6).float()
        y_pred = F.log_softmax(y_pred)
        loss = nn.NLLLoss2d(Variable(torch.FloatTensor([0.4,1, 1])).cuda())(y_pred, y)
        #loss = nn.CrossEntropyLoss(Variable(torch.FloatTensor([0.4,1, 1])).cuda())(y_pred, y)
        tot += loss

        if 0:
            X = X.data.squeeze(0).cpu().numpy()
            X = np.transpose(X, axes=[1, 2, 0])
            y = y.data.squeeze(0).cpu().numpy()
            y_pred = y_pred.data.squeeze(0).squeeze(0).cpu().numpy()
            print(y_pred.shape)

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 4, 1)
            ax1.imshow(X)
            ax2 = fig.add_subplot(1, 4, 2)
            ax2.imshow(y)
            ax3 = fig.add_subplot(1, 4, 3)
            ax3.imshow((y_pred > 0.5))

            Q = dense_crf(((X * 255).round()).astype(np.uint8), y_pred)
            ax4 = fig.add_subplot(1, 4, 4)
            print(Q)
            ax4.imshow(Q > 0.5)
            plt.show()
    return tot / i

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-n', '--model_name', dest='model_name',
                      default='deeplab', type=str, help='model_name')
    parser.add_option('-i', '--gpu_id', dest='gpu_id',
                      default='4', help='gpu_id')
    (options, args) = parser.parse_args()
    # resnet18 = deeplab_vgg16.vgg16_bn(pretrained=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_id

    #net = UNet(3, 3)
    net = deeplab_vgg16.vgg16_freespace_gn()
    print(net)

    net.load_state_dict(torch.load(options.load))
    print('Model loaded from {}'.format(options.load))
    net.eval()

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True
    dataset_dir = '/data0/hwkuang/data/Nullmax_data/'
    val_dataset_file = ops.join(dataset_dir, 'val_instance.txt')
    val_dataset = Dataset(val_dataset_file, dataset_dir, transform=SSDAugmentation())
    val_dataloader = data.DataLoader(val_dataset, 1, num_workers=1,
                                     shuffle=False, collate_fn=detection_collate, pin_memory=True)

    loss = my_eval_net(net, val_dataloader, options.gpu, options.model_name)
    print(loss)
