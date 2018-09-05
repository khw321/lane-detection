import sys
import os
from optparse import OptionParser
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
import os.path as ops
from eval import my_eval_net
from utils import data_processor
import torch.nn.init as init
from deeplab import deeplab_vgg16
from utils.pil_aug import SSDAugmentation
from dataset.nm_data import Dataset, detection_collate
import torch.utils.data as data
import math
from Instance_loss import Instance_loss
from matplotlib import pyplot as plt

# VGG_MEAN = [124, 117, 104]
VGG_MEAN = [92, 93, 93]


def train_net(net, epochs=5, batch_size=2, lr=0.1, val_percent=0.05,
              cp=True, gpu=False, model_name='deeplab', vis=False):
    if vis:
        vis_title = 'lane segmentation ' + model_name
        vis_legend = ['Loc Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    dir_checkpoint = 'experiment/deeplab/models/{}/'.format(model_name)
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    # dataset_dir = '/home/workspace/hwkuang/data/Nullmax_data/'
    # train_dataset_file = ops.join(dataset_dir, 'train_undistortion.txt')
    # val_dataset_file = ops.join(dataset_dir, 'val_undistortion.txt')
    dataset_dir = '/data0/hwkuang/data/Nullmax_data/'
    train_dataset_file = ops.join(dataset_dir, 'train_instance.txt')
    val_dataset_file = ops.join(dataset_dir, 'val_instance.txt')

    assert ops.exists(train_dataset_file)

    # train_dataset = data_processor.DataSet(train_dataset_file, dataset_dir)
    # val_dataset = data_processor.DataSet(val_dataset_file, dataset_dir)
    train_dataset = Dataset(train_dataset_file, dataset_dir, transform=SSDAugmentation())
    val_dataset = Dataset(val_dataset_file, dataset_dir, transform=SSDAugmentation())
    # a = train_dataset.pull_item(10)
    train_dataloader = data.DataLoader(train_dataset, batch_size, num_workers=4,
                                       shuffle=True, collate_fn=detection_collate, pin_memory=True)
    val_dataloader = data.DataLoader(val_dataset, 1, num_workers=1,
                                       shuffle=False, collate_fn=detection_collate, pin_memory=True)

    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    if gpu:
        weight = Variable(torch.FloatTensor([0.4, 1])).cuda()
    else:
        weight = Variable(torch.FloatTensor([0.4, 1]))

    lr_step_index = 0
    lr_step = [10, 20]
    criterion = nn.NLLLoss2d(weight)
    discriminative_loss = Instance_loss()
    epoch_size = len(train_dataloader)
    global_step = 0
    for epoch in range(epochs):
        print('\nEpoch: {}'.format(epoch))
        net.train()
        start_batch_idx = len(train_dataloader) * epoch
        epoch_loss = 0
        # if epoch in lr_step:
        #     lr_step_index += 1
        #     adjust_learning_rate(optimizer, 0.1, lr_step_index)

        for batch_idx,  (inputs, targets) in enumerate(train_dataloader):
            global_step = batch_idx + start_batch_idx
            batch_lr = lr * sgdr(len(train_dataloader) * 20, global_step)
            adjust_learning_rate_every_epoch(optimizer, batch_lr)
            # adjust_learning_rate_every_epoch(optimizer, lr * (global_step / (len(train_dataloader) * epochs)))

            optimizer.zero_grad()
            Ins_img = targets.numpy()
            Bi_img = 1*(Ins_img>0)
            y2 = torch.from_numpy(Bi_img)
            # y1 = torch.unsqueeze(targets, 1)
            if gpu:
                X = Variable(inputs).cuda()
                y1 = Variable(targets).cuda()
                y2 = Variable(y2).cuda()
            else:
                X = Variable(inputs)
                y1 = Variable(targets)
                y2 = Variable(y2)

            y_pred = net(X)
            yp1 = y_pred[:, :2, :, :]
            yp2 = y_pred[:, 2:, :, :]
            prob1 = F.log_softmax(yp1)
            prob2 = F.log_softmax(yp2)
            loss1 = discriminative_loss(prob1, y1)
            loss2 = criterion(prob2, y2)
            loss = 0.2 * loss1 + loss2
            loss.backward()
            optimizer.step()

            epoch_loss += loss.data[0]

            if global_step % 100 == 0:
                data_processor.draw_fs(X, y1, y2, F.softmax(prob1), F.softmax(prob2), global_step, 'mid_result/{}/train/'.format(model_name))

            if global_step % 10 == 0:
                print('{}-{} --- loss: {} , loss_dist: {} , loss_bi: {} --- learning rate: {:6f}'.format(
                    epoch, batch_idx, loss.data[0], loss1.data[0], loss2.data[0], batch_lr))

            if vis:
                res = [1, loss.data[0]][loss.data[0]<1]
                update_vis_plot(global_step, res, iter_plot, 'append')

            # if batch_idx and batch_idx % (epoch_size // 2 - 1) == 0:
            #     print('Half Epoch {} -- loss: {}'.format(epoch, epoch_loss))
            #     epoch_loss = 0
            #     val_loss = my_eval_net(net, val_dataloader, gpu, model_name)
            #     print('val Loss:{}'.format(val_loss))
            #     torch.save(net.state_dict(),
            #                dir_checkpoint + 'CP{}_{}.pth'.format(epoch, batch_idx))
            #     print('Checkpoint {} saved !'.format(global_step + 1))
        # if epoch and epoch % (epoch_size // 2 - 1) == 0:
        #     print('Half Epoch {} -- loss: {}'.format(epoch, epoch_loss))
        #     epoch_loss = 0
        #     val_loss = my_eval_net(net, val_dataloader, gpu, model_name)
        #     print('val Loss:{}'.format(val_loss))
        torch.save(net.state_dict(),
                   dir_checkpoint + 'CP{}_{}.pth'.format(epoch, global_step))
        print('Checkpoint {} saved !'.format(epoch))
def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 1)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loss, window1, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 1)).cpu() * iteration,
        Y=torch.Tensor([loss]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )

def xavier(param):
    init.xavier_uniform(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = 0.01 * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate_every_epoch(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_optimizer_lr(optimizer, lr):
    # callback to set the learning rate in an optimizer, without rebuilding the whole optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

def sgdr(period, batch_idx):
    # returns normalised anytime sgdr schedule given period and batch_idx
    # best performing settings reported in paper are T_0 = 10, T_mult=2
    # so always use T_mult=2
    batch_idx = float(batch_idx)
    restart_period = period
    while batch_idx/restart_period > 1.:
        batch_idx = batch_idx - restart_period
        restart_period = restart_period * 2.

    radians = math.pi*(batch_idx/restart_period)
    return 0.5*(1.0 + math.cos(radians))

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=20, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-v', '--load_vgg', dest='load_vgg',
                      default=False, help='load vgg model')
    parser.add_option('-n', '--model_name', dest='model_name',
                      default='deeplab', type=str, help='the prefix name of save files')
    parser.add_option('-i', '--gpu_id', dest='gpu_id',
                      default='5', help='gpu_id')
    parser.add_option('--visdom', default=False, type=str,
                        help='Use visdom for loss visualization')
    (options, args) = parser.parse_args()
    # resnet18 = deeplab_vgg16.vgg16_bn(pretrained=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu_id
    print(options)
    net = deeplab_vgg16.vgg16_freespace_gn()
    print(net)
    # tin = torch.randn([2,3,480,640])
    # tout = net(Variable(tin))

    # load pretrained vgg model
    if options.load_vgg:
        print('load vgg')
        pretrained_dict = torch.load('mid_result/vgg16_bn-6c64b313.pth')
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)

    # create visdom for loss plot
    if options.visdom:
        import visdom
        viz = visdom.Visdom(use_incoming_socket=False)

    # # xavier init
    # net.apply(weights_init)

    # load model
    if options.load:
        pretrained_dict = torch.load(options.load)
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print('Model loaded from {}'.format(options.load))

        # net.load_state_dict(torch.load(options.load))

    if options.gpu:
        net.cuda()
        cudnn.benchmark = True

    try:
        train_net(net, options.epochs, options.batchsize, options.lr,
                  gpu=options.gpu, model_name=options.model_name, vis=options.visdom)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
