from train import train
import evals
import data_loader

import chainer
from chainer import serializers

import copy
import json
#import pickle
import os
#from tqdm import tqdm


def get_best_model_by_accuracy(settings, max_ittr, xs_val, ts_val, zs_val, word2vec, out, snapshot_interval):
    best = {'ittr':-1, 
            'val@1':-1, 
            'history':[]}
    for i in range(snapshot_interval, max_ittr, snapshot_interval):
        net = copy.deepcopy(settings['model']['net']) 
        serializers.load_npz('{}/net_iter_{}.npz'.format(out, i), net)
        result = evals.get_accuracy(xs_val, ts_val, zs_val, net, word2vec, n_top=[1], show=False)
        acc_1 = result['mean_acc@1']
        best['history'].append({'ittr':i, 'val@1':acc_1})
        print('ittr {} val@1: {}'.format(i, acc_1))
        if best['val@1']<acc_1:
            best['val@1'] = acc_1
            best['ittr'] = i
    print('best ittr: ', best['ittr'])
    print('best acc  : ', best['val@1'])
    return best


def get_best_model_by_loss(settings, max_ittr, xs_val, ts_val, zs_val, word2vec, out, snapshot_interval):
    best = {'ittr':-1, 
            'val_loss':10000, 
            'history':[]}
    for i in range(snapshot_interval, max_ittr, snapshot_interval):
        net = copy.deepcopy(settings['model']['net']) 
        serializers.load_npz('{}/net_iter_{}.npz'.format(out, i), net)
        #result = evals.get_accuracy(xs_val, ts_val, zs_val, net, word2vec, n_top=[1], show=False)
        val_loss = evals.get_val_loss(xs_val, ts_val, zs_val, net, word2vec)
        best['history'].append({'ittr':i, 'val_loss':val_loss})
        print('ittr {} val loss: {}'.format(i, val_loss))
        if best['val_loss'] > val_loss:
            best['val_loss'] = val_loss
            best['ittr'] = i
    print('best ittr: ', best['ittr'])
    print('best loss  : ', best['val_loss'])
    return best


def experiment1(settings):
    experiment_name = settings['experiment_name']
    word2vec = settings['word2vec']
    num_N_train = settings['num_N_train']
    n_argument = settings['n_argument']
    batchsize = settings['batchsize'] 
    epoch = settings['epoch'] 
    alpha = settings['alpha'] 
    datasets = settings['datasets'] 
    path_dataset = settings['path_dataset'] 
    n_attribute = settings['n_attribute'] 
    num_N_test = settings['num_N_test'] 
    n_top = settings['n_top'] 
    gpu = settings['gpu'] 
    net = copy.deepcopy(settings['model']['net']) 
    Updater = copy.deepcopy(settings['updater']['updater']) 
    out = settings['out'] 
    resume = settings['resume'] 
    snapshot_interval = settings['snapshot_interval'] 
    display_interval = settings['display_interval']
    emb_method = settings['embedding_method'] # word2vec, glove


    # Load data
    _xs_train, xs_val, xs_test, _ts_train, ts_val, ts_test, _zs_train, \
    zs_val, zs_test, M, F, xs, ts, zs, A_train, A_val, A_test, z_ids, \
    N_train = data_loader.load_datasets(datasets, word2vec, num_N_train, path_dataset, emb_method)

    
    # Data argumention
    xs_train = data_loader.data_argument(_xs_train, n_argument)
    ts_train = data_loader.data_argument(_ts_train, n_argument)
    zs_train = data_loader.data_argument(_zs_train, n_argument)
    #xs_train = _xs_train
    #ts_train = _ts_train
    #zs_train = _zs_train
        
        
    # Train
    if settings['post_process']['train']:
        train(net, 
              Updater, 
              word2vec, 
              batchsize, 
              epoch, 
              xs_train,
              ts_train,
              zs_train,
              xs_val,
              ts_val,
              zs_val,
              out, 
              snapshot_interval, 
              display_interval,
              resume=resume,
              alpha=alpha,
              gpu=gpu) 
    else: # Evaluation only
        settings['experiment_name'] = resume.split('/')[-2]
        settings['out'] = resume
        out = settings['out']

    
    # Get a best model
    if settings['post_process']['best_model']:
        print('\nSearching best model...')
        max_ittr = epoch * (len(xs_train)//batchsize)
        #best = get_best_model_by_accuracy(settings, max_ittr, xs_val, ts_val, zs_val, word2vec, out, snapshot_interval)
        best = get_best_model_by_accuracy(settings, max_ittr, xs_val, ts_val, zs_val, word2vec, out, snapshot_interval)
        net = copy.deepcopy(settings['model']['net'])
        best_path = '{}/net_iter_{}.npz'.format(out, best['ittr'])
        #best_path = '{}/{}'
        serializers.load_npz(best_path, net)
    else:
        best = ''
        
    
    # Evaluation (Accuracy)
    if settings['post_process']['acc']:
        print('\ncalculating accuracies...')
        train_results = {}
        val_results = {}
        test_results = {}
        for z_id in set(zs): 
            print('\tz: ', z_ids[str(z_id)])
            train_results[str(z_id)] = evals.evaluation1_acc(A_train[z_id]['xs'], A_train[z_id]['ts'], \
                                                             A_train[z_id]['zs'], net, word2vec, n_top, 'train')
            val_results[str(z_id)] = evals.evaluation1_acc(A_val[z_id]['xs'], A_val[z_id]['ts'], \
                                                           A_val[z_id]['zs'], net, word2vec, n_top, 'val')
            test_results[str(z_id)] = evals.evaluation1_acc(A_test[z_id]['xs'], A_test[z_id]['ts'], \
                                                            A_test[z_id]['zs'], net, word2vec, n_top, 'test')
    else:
        train_results, val_results, test_results = '', '', ''

        
    # Evaluation (Stability)
    if settings['post_process']['stb']:
        print('\ncalculating stability score...')
        stb_results = {}
        for z_id in set(zs): 
            print('\tz: ', z_ids[str(z_id)])
            stb_results[str(z_id)] = evals.evaluation1_stb(net, z_id, word2vec, M, F, num_N_test, n_top)
    else:
        stb_results = ''
    
    
    # Save results
    if settings['post_process']['save_results']:
        print('\nsaving results and experimental settings...')
        del settings['word2vec']
        del settings['model']['net']
        del settings['updater']['updater']
        results = {}
        results['experimental_settings'] = settings
        results['net'] = {'best':best_path, 
                          'history':best}
        results['accuracy'] = {'train':train_results, 
                               'val':val_results, 
                               'test':test_results}
        results['stability'] = {'score':stb_results, 
                                'num_N_test':{'size':num_N_test,
                                              'data':stb_results[str(0)]['x']}, 
                                'num_N_train':{'size':num_N_train,
                                               'data':N_train}}
        path_save = '{}/evaluation_results'.format('/'.join(out.split('/')[:-1]))
        os.makedirs(path_save, exist_ok=True)
        with open('{}/evaluation_results.json'.format(path_save), 'w') as f:
            json.dump(results, f)

    print('done.')
