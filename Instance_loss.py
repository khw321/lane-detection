#
# myloss.py : implementation of the Dice coeff and the associated loss
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



class Instance_loss(nn.Module):
    """
    input: batch x 2 x h x w
    target: batch x h x w
    """

    def __init__(self, delta_var=0.5, delta_dist=1.5, Lnorm=2):
        super().__init__()

        self.delta_var = delta_var
        self.delta_dist = delta_dist

    def forward(self, inputs, targets):
        b, c, h, w = inputs.shape
        # nInstance = targets.shape[1]
        nInstance = 1
        loss = Variable(torch.zeros(1)).cuda()
        for i in range(b):
            pred = inputs[i]  # the numble i image
            loss_var = 0.0
            loss_dist = 0.0

            for j in range(nInstance):  # catagory class number, for lane, nInstance=1
                target = targets[i].view(1, h, w)  # the i image, j class
                means = []
                loss_v = 0.0
                loss_d = 0.0

                # center pull force
                max_id = torch.max(target.data)  # lanes' number  --------------C
                for l in range(1, int(max_id)+1):  # for every lane
                    mask = target.eq(l)  # one lane's mask, 1 x h x w
                    mask_sum = torch.sum(mask.data)  # -------------------------Nc
                    if mask_sum > 1:
                        # mask.expand_as(pred).shape = pred.shape = c x h x w
                        # pred[mask.expand_as(pred)].shape = c*h*w
                        # .view(c, -1, 1).shape = c x h*w x 1
                        inst = pred[mask.expand_as(pred)].view(c, -1, 1)  #

                        # Calculate mean of instance
                        #k torch.mean(inst, 1).shape = c x 1
                        mean = torch.mean(inst, 1).view(c, 1, 1)  # c x 1 x 1
                        means.append(mean)

                        # Calculate variance of instance
                        var = self.norm((inst - mean.expand_as(inst)), 2)  # 1 x -1 x 1, the right 2 means Ln distance
                        var = torch.clamp(var - self.delta_var, min=0.0)

                        var = torch.pow(var, 2)
                        var = var.view(-1)

                        var = torch.mean(var)
                        loss_v = loss_v + var

                loss_var = loss_var + loss_v

                # center push force
                if len(means) > 1:
                    for m in range(0, len(means)):
                        mean_A = means[m]  # c x 1 x 1
                        for n in range(m + 1, len(means)):
                            mean_B = means[n]  # c x 1 x 1
                            d = self.norm(mean_A - mean_B, 2)  # 1 x 1 x 1
                            d = torch.pow(torch.clamp(-(d - 2 * self.delta_var), min=0.0), 2)
                            loss_d = loss_d + d[0][0][0]

                    loss_dist = loss_dist + loss_d / ((len(means) - 1) + 1e-8)

            loss = loss + (loss_dist + loss_var)

        loss = loss / b    # + torch.sum(inputs) * 0

        return loss

    def norm(self, inp, L):
        if L == 1:
            n = torch.sum(torch.abs(inp), 0)
        else:
            n = torch.sqrt(torch.sum(torch.pow(inp, 2), 0) + 1e-8)
        return n
if __name__ == '__main__':
    input_tensor = torch.randn([2, 2, 10, 20])
    target = torch.ones([2, 1, 10, 20])
    loss = Instance_loss()
    loss(Variable(input_tensor), Variable(target))