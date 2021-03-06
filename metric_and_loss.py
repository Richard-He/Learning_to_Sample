import torch.nn as nn
import torch


class NormCrossEntropyLoss(object):
    def __init__(self):
        self.loss_op = nn.CrossEntropyLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.long())
        loss = (loss * data.node_norm)
        return loss


class NormBCEWithLogitsLoss(object):
    def __init__(self):
        self.loss_op = nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, out, data):
        loss = self.loss_op(out, data.y.type_as(out))
        #print('loss shape: ',loss.shape)
        #print('data node norm shape: ',data.node_norm.shape)
        # loss = (loss * data.node_norm)[data.train_mask].sum()
        loss = torch.mul(loss.T, data.node_norm).T
        return loss
