import sys
import os
from optparse import OptionParser
#matplotlib.use('Agg')
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

# VGG_MEAN = [124, 117, 104]
VGG_MEAN = [92, 93, 93]


def train_net(net, epochs=5, batch_size=2, lr=0.1, val_percent=0.05,
              cp=True, gpu=False, model_name='deeplab', vis=False):
    if vis:
        vis_title = 'lane and freespace segmentation ' + model_name
        vis_legend = ['Loc Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)

    dir_checkpoint = 'experiment/deeplab/models/{}/'.format(model_name)
    if not os.path.exists(dir_checkpoint):
        os.makedirs(dir_checkpoint)
    dataset_dir = '/home/workspace/hwkuang/data/Nullmax_data/'
    train_dataset_file = ops.join(dataset_dir, 'train_undistortion.txt')
    val_dataset_file = ops.join(dataset_dir, 'val_undistortion.txt')

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
    batch_iterator = iter(train_dataloader)

    # optimizer = optim.Adam(net.parameters(), lr=lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    if gpu:
        weight = Variable(torch.FloatTensor([0.4, 1])).cuda()
    else:
        weight = Variable(torch.FloatTensor([0.4, 1]))

    criterion = nn.NLLLoss2d(weight)
    cfg_lr = [20000, 30000, 35000]
    step_index = 0
    val_max = [1, 10]
    epoch_loss = 0
    epoch_size = len(train_dataset) // batch_size
    for epoch in range(1, epochs):

        # if epoch in cfg_lr:
        #     step_index += 1
        #     adjust_learning_rate(optimizer, 0.01, step_index)

        adjust_learning_rate_every_epoch(optimizer, lr*(1-epoch/epochs))
        try:
            img, lane, fs = next(batch_iterator)
        except StopIteration:
            print('Starting epoch {}/{}.'.format(epoch // epoch_size, epochs // epoch_size))
            batch_iterator = iter(train_dataloader)
            img, lane, fs = next(batch_iterator)

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
        prob1 = F.log_softmax(yp1)
        prob2 = F.log_softmax(yp2)

        loss1 = criterion(prob1, y1)
        loss2 = nn.NLLLoss2d()(prob2, y2)
        loss = loss1 + loss2
        epoch_loss += loss.data[0]

        optimizer.zero_grad()
        loss.backward()

        if epoch % 10 == 0:

            print('{} --- loss: {}, loss1: {} , loss2: {}'.format(epoch, loss.data[0], loss1.data[0], loss2.data[0]))
        if vis:
            update_vis_plot(epoch, loss.data[0],
                            iter_plot, 'append')

        if epoch % 100 == 0:
            data_processor.draw_fs(X, y1, y2, F.softmax(prob1), F.softmax(prob2), epoch, 'mid_result/{}/train/'.format(model_name))

        optimizer.step()
        if epoch % 1 == 0:
            print('Starting epoch {}.'.format(epoch//2000))
            print('Epoch finished ! Loss: {}'.format(epoch_loss / 2000))

            epoch_loss = 0
            if epoch % 1 == 0:
                val_loss = my_eval_net(net, val_dataloader, gpu, model_name)
                print('val Loss:{}'.format(val_loss))
            # if cp:
                torch.save(net.state_dict(),
                           dir_checkpoint + 'CP{}.pth'.format(epoch + 1))

                print('Checkpoint {} saved !'.format(epoch + 1))

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



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=80000, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=2,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-v', '--load_vgg', dest='load_vgg',
                      default=True, help='load vgg model')
    parser.add_option('-n', '--model_name', dest='model_name',
                      default='deeplab', type=str, help='the prefix name of save files')
    parser.add_option('-i', '--gpu_id', dest='gpu_id',
                      default='0', help='gpu_id')
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
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

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
