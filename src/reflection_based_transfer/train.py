import numpy
import argparse
import os
import json

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainer import Variable


# Reset numpy / cupy seed
def reset_seed(seed):
    #random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
        
        

def get_z(z_id, dim_z):
    numpy.random.seed(z_id) 
    return numpy.random.uniform(-1, 1, (dim_z,)).astype(numpy.float32)
        
    

class AttributeTransferDataset(chainer.dataset.DatasetMixin):
    
    def __init__(self, xs, zs, ts, word2vec, dtype=numpy.float32):
        self._dtype = dtype
        self._xs = xs # ['man', 'boy', ...]
        self._zs = zs # [0, 0, ...]
        self._ts = ts # ['woman', 'girl', ...]
        self._word2vec = word2vec

    def __len__(self):
        return len(self._xs)

    def get_example(self, i):
        # Return word vector x, attribute vector z, target word vector t
        x = self._word2vec[self._xs[i]].astype(numpy.float32)
        t = self._word2vec[self._ts[i]].astype(numpy.float32)
        z = self._zs[i] # int e.g., 0
        return x, z, t
    


def train(net, Updater, word2vec, batchsize, epoch, \
          xs_train, ts_train, zs_train, xs_val, ts_val, zs_val, out, snapshot_interval, \
          display_interval, resume='', gpu=-1, alpha=-0.0002, beta1=0.5, seed=0):
    
    reset_seed(seed)
    
    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        net.to_gpu()  # Copy the model to the GPU
    
    # Setup an optimizer
    #def make_optimizer(model, alpha=0.0002, beta1=0.5):
    def make_optimizer(model, alpha=alpha, beta1=beta1):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        #optimizer.add_hook( 
        #    chainer.optimizer_hooks.WeightDecay(0.0001), 'hook_dec')
        return optimizer
    
    opt = make_optimizer(net)
    
    # Setup dataset
    train = AttributeTransferDataset(xs_train, zs_train, ts_train, word2vec)
    val   = AttributeTransferDataset(xs_val, zs_val, ts_val, word2vec)
    
    # Setup an iterator
    train_iter = chainer.iterators.SerialIterator(train, batchsize) # shuffle=True)
    val_iter   = chainer.iterators.SerialIterator(val, batchsize, False, False)#shuffle=False, repeat=False) # batchsize=len(val)
    
    # Setup an updater
    updater = Updater(model=net,
                      iterator=train_iter,
                      optimizer={'net': opt},
                      device=gpu)
    
    # Setup a trainer
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    
    snapshot_interval = (snapshot_interval, 'iteration')
    display_interval = (display_interval, 'iteration')
    trainer.extend(
        extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),
        trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        net, 'net_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
        
    #trainer.extend(extensions.ProgressBar(update_interval=10))
    
    #trainer.extend(extensions.Evaluator(val_iter, net, device=gpu, eval_func=net.eval_func), name='val')
    
    #trainer.extend(extensions.PrintReport([
    #    'epoch', 'iteration', 'net/loss', 'val/net/loss'
    #]), trigger=display_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'net/loss',
    ]), trigger=display_interval)
    
    #trainer.extend(extensions.PlotReport(['net/loss', 'validation/net/loss'], \
    #                                     x_key='epoch', file_name='loss.png'))
    
    if resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(resume, trainer)
    
    # Run the training
    trainer.run()

