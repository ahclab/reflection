import json
import random
import numpy
import chainer
import copy
from tqdm import tqdm


def reset_seed(seed):
    #random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
        
        
def data_argument(word_list, n_argument):
    copy_list = copy.deepcopy(word_list)
    for i in range(n_argument):
        if type(numpy.array([1])) is numpy.ndarray:
            word_list = numpy.append(word_list, copy_list)
        else:
            word_list.append(copy_list)
    return word_list



def load_attribute_words(datasets, path_dataset, embedding_method):
    
    xs_train, xs_val, xs_test, ts_train, ts_val, ts_test, \
    zs_train, zs_val, zs_test, M, F, xs, ts, zs \
    = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    
    A_train, A_val, A_test, z_ids = {}, {}, {}, {}
    
    # Mix attribute words
    for attr_id, dset in enumerate(datasets):
        z_ids[str(attr_id)] = dset 
        
        # Lad a dataset
        with open('{}/{}_{}.json'.format(path_dataset, dset, embedding_method)) as f:
            dataset = json.load(f)
            dataset = dataset['dataset']
        
        # Add data to lists
        xs       += dataset['xs']
        xs_train += dataset['xs_train']
        xs_val   += dataset['xs_val']
        xs_test  += dataset['xs_test']
        ts       += dataset['ts']
        ts_train += dataset['ts_train']
        ts_val   += dataset['ts_val']
        ts_test  += dataset['ts_test']
        zs       += [attr_id for _ in dataset['zs']]
        zs_train += [attr_id for _ in dataset['zs_train']]
        zs_val   += [attr_id for _ in dataset['zs_val']]
        zs_test  += [attr_id for _ in dataset['zs_test']]
        M        += dataset['M']
        F        += dataset['F']
        
        A_train[attr_id] = {'xs':dataset['xs_train'], 'ts':dataset['ts_train'], \
                            'zs':numpy.asarray(dataset['zs_train']).astype(numpy.int32)}
        A_val[attr_id] = {'xs':dataset['xs_val'], 'ts':dataset['ts_val'], \
                          'zs':numpy.asarray(dataset['zs_val']).astype(numpy.int32)}
        A_test[attr_id] = {'xs':dataset['xs_test'], 'ts':dataset['ts_test'],\
                           'zs':numpy.asarray(dataset['zs_test']).astype(numpy.int32)}
        
    '''
    # Shuffle all pairs (複数の属性ペアがあるときに必要。)
    def shuffle_data(x,t,z):
        d_index = list(range(len(x)))    
        random.seed(0) # seed = 0 固定
        d_index = random.sample(d_index, len(d_index))
        return [x[i] for i in d_index], [t[i] for i in d_index], [z[i] for i in d_index]

    xs_train, ts_train, zs_train = shuffle_data(xs_train, ts_train, zs_train)
    xs_val, ts_val, zs_val = shuffle_data(xs_val, ts_val, zs_val)
    xs_test, ts_test, zs_test = shuffle_data(xs_test, ts_test, zs_test)
    '''

    zs = numpy.asarray(zs).astype(numpy.int32)
    zs_train = numpy.asarray(zs_train).astype(numpy.int32)
    zs_val = numpy.asarray(zs_val).astype(numpy.int32)
    zs_test = numpy.asarray(zs_test).astype(numpy.int32)
    assert zs.shape == (len(xs), ), zs.shape
    return xs_train, xs_val, xs_test, ts_train, ts_val, ts_test, zs_train, \
           zs_val, zs_test, M, F, xs, ts, zs, A_train, A_val, A_test, z_ids



def get_non_attribute_words(num_N, M, F, vocab):
    '''
        num_N: num of non-attribute words
    '''
    N = []
    n = copy.deepcopy(num_N)
    i = 0
    while(len(N)!=num_N):
        random.seed(i)
        non_attribute_words = random.sample(vocab, n) # ['apple', 'aaa', ...]
        for nw in non_attribute_words:
            if nw not in N + M + F:
                N.append(nw)
        n = num_N - len(N)
        i += 1
    return N



def load_datasets(datasets, word2vec, num_sampling, path_dataset, embedding_method):
    '''
        e.g.
        datasets = ['capital-country_109', 'male-female_101']
        path_dataset = '../../data/datasets'
        embedding_method = 'word2vec'
    '''
    print('Loading dataset...')    
    xs_train, xs_val, xs_test, ts_train, ts_val, ts_test, \
    zs_train, zs_val, zs_test, M, F, xs, ts, zs, A_train, \
    A_val, A_test, z_ids = load_attribute_words(datasets, path_dataset, embedding_method)
    print('Dataset was loaded.')
    
    # Negative sampling
    if num_sampling: # n_sampling != 0
        vocab = list(word2vec.vocab.keys())
        vocab.sort() 
        print('Sampling non-attribute words from the vocabulary...')    
        N_train = get_non_attribute_words(num_sampling, M, F, vocab)
        xs_train += N_train 
        ts_train += N_train
        for z in set(zs):
            for i in range(num_sampling) :
                zs_train = numpy.append(zs_train, z)
        print('Sampling done.')
    else:
        N_train = []

    return xs_train, xs_val, xs_test, ts_train, ts_val, ts_test, zs_train, \
           zs_val, zs_test, M, F, xs, ts, zs, A_train, A_val, A_test, z_ids, N_train


if __name__ == "__main__":
    datasets = ['capital-country_109', 'male-female_101']
    path_dataset = '../../data/datasets'
    embedding_method = 'word2vec'
    xs_train, xs_val, xs_test, ts_train, ts_val, \
    ts_test, zs_train, zs_val, zs_test, M, F, xs, \
    ts, zs, A_train, A_val, A_test, z_ids = load_attribute_words(datasets, path_dataset, embedding_method)
    
