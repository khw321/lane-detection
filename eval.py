import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import time
VGG_MEAN = [123.68, 116.779, 103.939]
from utils.pil_aug import SSDAugmentation

import os
from optparse import OptionParser

import torch.backends.cudnn as cudnn
import os.path as ops
from utils import *
from utils import data_processor
from deeplab import deeplab_vgg16
from dataset.nm_data import Dataset, detection_collate
import torch.utils.data as data

def my_eval_net(net, dataset, gpu=False, *args):
    tot = 0
    num_images = len(dataset)
    t1 = time.time()
    data = iter(dataset)
    for i in range(num_images):
        img, lane, fs = next(data)

        # imgs = []
        # lanes = []
        # fss = []
        # for i in range(1):
        #     imgs.append(torch.from_numpy(img.transpose(2,0,1)))
        #     lanes.append(torch.from_numpy(lane))
        #     fss.append(torch.from_numpy(fs))
        # X, y_lane, y_fs = torch.stack(imgs, 0), torch.stack(lanes, 0).long(), torch.stack(fss, 0).long()

        if gpu:
            X = Variable(img).cuda()
            y1 = Variable(lane).cuda()
            y2 = Variable(fs).cuda()
        else:
            X = Variable(img)
            y1 = Variable(lane)
            y2 = Variable(fs)

        y_pred = net(X)
        yp1 = y_pred[:, :2, :, :]
        yp2 = y_pred[:, 2:, :, :]

        # draw mid_result
        if i % 10 == 0:
            data_processor.draw_fs(X, y1, y2, F.softmax(yp1), F.softmax(yp2), i, 'mid_result/{}/val/{}/'.format(args[0], t1))


        prob1 = F.log_softmax(yp1)
        prob2 = F.log_softmax(yp2)

        loss1 = nn.NLLLoss2d(Variable(torch.FloatTensor([0.4,1])).cuda())(prob1, y1)
        loss2 = nn.NLLLoss2d()(prob2, y2)
        loss = loss1 + loss2

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
                      default='0', help='gpu_id')
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
    dataset_dir = '/home/workspace/hwkuang/data/Nullmax_data/'
    val_dataset_file = ops.join(dataset_dir, 'val_undistortion.txt')
    val_dataset = Dataset(val_dataset_file, dataset_dir, transform=SSDAugmentation())
    val_dataloader = data.DataLoader(val_dataset, 1, num_workers=1,
                                     shuffle=False, collate_fn=detection_collate, pin_memory=True)

    loss = my_eval_net(net, val_dataloader, options.gpu, options.model_name)
    print(loss)
