import os
import argparse
import time
import numpy
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import cv2
from unet import UNet
from utils import *
from matplotlib import pyplot as plt
import torch.nn as nn

VGG_MEAN = [123.68, 116.779, 103.939]

def yuanshi_predict_img(net, full_img, gpu=False):
    img = resize_and_crop(full_img)

    left = get_square(img, 0)
    right = get_square(img, 1)

    right = normalize(right)
    left = normalize(left)

    right = np.transpose(right, axes=[2, 0, 1])
    left = np.transpose(left, axes=[2, 0, 1])

    X_l = torch.FloatTensor(left).unsqueeze(0)
    X_r = torch.FloatTensor(right).unsqueeze(0)

    if gpu:
        X_l = Variable(X_l, volatile=True).cuda()
        X_r = Variable(X_r, volatile=True).cuda()
    else:
        X_l = Variable(X_l, volatile=True)
        X_r = Variable(X_r, volatile=True)

    y_l = F.sigmoid(net(X_l))
    time_start = time.time()

    y_r = F.sigmoid(net(X_r))
    time_stop = time.time()
    print('time----{}'.format(time_stop - time_start))

    y_l = F.upsample_bilinear(y_l, scale_factor=2).data[0][0].cpu().numpy()
    y_r = F.upsample_bilinear(y_r, scale_factor=2).data[0][0].cpu().numpy()

    y = merge_masks(y_l, y_r, full_img.size[0])
    return y
    # yy = dense_crf(np.array(full_img).astype(np.uint8), y)

   # return yy > 0.5
    return yy
def predict_img(net, full_img, gpu=False):
    img = full_img.resize([640,480])
    img = np.array(img) - VGG_MEAN

    # img = normalize(img)

    img = np.transpose(img, axes=[2, 0, 1])

    X_r = torch.FloatTensor(img).unsqueeze(0)

    if gpu:
        X_r = Variable(X_r, volatile=True).cuda()
    else:
        X_r = Variable(X_r, volatile=True)


    time_start = time.time()

    y_r = F.softmax(net(X_r))
    # y_r = net(X_r)
    time_stop = time.time()
    print('time----{}'.format(time_stop - time_start))

    #y_r = F.upsample_bilinear(y_r, scale_factor=2).data[0][0].cpu().numpy()
    y_r = y_r[0].data.cpu().numpy()
    # y_r = F.upsample_bilinear(y_r, scale_factor=2).data[0][0].cpu().numpy()
    y_r1 = cv2.resize(y_r[1], (640, 480))
    y_r2 = cv2.resize(y_r[2], (640, 480))
    return y_r1, y_r2
    yy = dense_crf(np.array(full_img).astype(np.uint8), y)

   # return yy > 0.5
    return yy


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='experiment/up1/models/scnn_xavier/CP19201.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                             " (default : 'MODEL.pth')");
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='filenames of ouput images')
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_false',
                        help="Do not save the output masks",
                        default=False)

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 3)
    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
    else:
        net.cpu()
        print("Using CPU version of the net, this may be very slow")


    #in_files = args.input
    in_files = open('data/test.txt').readlines()
    out_files = []
    save_path = 'experiment/up1/predict/scnn_xaiver_out'
    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        print("Error : Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    print("Model loaded !")

    for i, fn in enumerate(in_files):
        print("\nPredicting image {} ...".format(fn))
        img = Image.open(fn[:-1])

        out, out2 = predict_img(net, img, not args.cpu)
        if args.viz:
            print("Vizualising results for image {}, close to continue ..."
                  .format(fn))

            fig = plt.figure()
            a = fig.add_subplot(1, 2, 1)
            a.set_title('Input image')
            plt.imshow(img)

            b = fig.add_subplot(1, 2, 2)
            b.set_title('Output mask')
            plt.imshow(out)

            plt.show()

        if not args.no_save:
            result1 = Image.fromarray((out * 100).astype(numpy.uint8))
            result2 = Image.fromarray((out2 * 100).astype(numpy.uint8))
            if not os.path.exists(os.path.dirname(os.path.join(save_path,out_files[i][:-1]))):
                os.makedirs(os.path.dirname(os.path.join(save_path,out_files[i][:-1])))
            result1.save(os.path.join(save_path,out_files[i][:-1]).replace('.jpg', '_1.jpg'))
            # result2.save(os.path.join(save_path,out_files[i][:-1]).replace('.jpg', '_2.jpg'))
            print("Mask saved to {}".format(out_files[i]))