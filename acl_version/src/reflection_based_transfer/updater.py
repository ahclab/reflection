#!/usr/bin/env python
import chainer
import chainer.functions as F
from chainer import Variable


class AttributeTransferUpdater1(chainer.training.updaters.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.net = kwargs.pop('model')
        super(AttributeTransferUpdater1, self).__init__(*args, **kwargs)

    def loss(self, net, y, t):
        loss = F.mean_squared_error(y, t)
        chainer.report({'loss': loss}, net)
        return loss

    def update_core(self):
        # Optimizer
        optimizer = self.get_optimizer('net')

        # Attribute Transfer
        net = self.net
        
        # Load data
        batch = self.get_iterator('main').next()
        batch = self.converter(batch, self.device)
        
        # Word vector
        x = Variable(batch[0])

        # Attribute
        z = Variable(batch[1])
        
        # Target word vector
        t = Variable(batch[2])

        # Transform a word vector x to a word vector y
        y = net(x, z)

        # Update
        optimizer.update(self.loss, net, y, t)
        

